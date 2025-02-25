import time
import psutil
import pulp
import torch

from dataclasses import dataclass
from pulp import LpProblem, LpVariable
from tqdm import tqdm
from typing import List

from flextrain.checkpointing import detach_variable, retrieve_tensor
from flextrain.config import get_flextrain_config
from flextrain.defaults import (
    BENCHMARK_GPU_ITERATIONS,
    BENCHMARK_PCIE_ITERATIONS,
    BENCHMARK_CPU_ITERATIONS,
    BENCHMARK_NVME_ITERATIONS,
    BENCHMARK_NUM_BLOCKS,
    BENCHMARK_PCIE_BLOCK_SIZE,
    BENCHMARK_NVME_BLOCK_SIZE,
    BENCHMARK_OPTIMIZER_BLOCK_SIZE,
    ACCESSIBLE_GPU_MEMORY_RATIO,
    ACCESSIBLE_CPU_MEMORY_RATIO,
    POTENTIAL_ALPHA_VALUES,
    THROUGHPUT_STABLE_THRESHOLD,
    REGULARIZATION_PENALTY,
    NUM_EXTRA_CONFIGS
)
from flextrain.llm_func import LLMFunc, retrieve_llm_loss
from flextrain.memory import FlexTrainDataType, get_allocated_dram_size
from flextrain.memory.coordinator import (
    get_para_coordinator, get_interlayer_coordinator, get_opts_coordinator
)
from flextrain.memory.nvme_swapper import get_nvme_swapper, FlexTrainDataID
from flextrain.utils import dist


def _save_vram_usage_estimation():
    """ Estimate the memory usage. Return the total memory in bytes. """
    free, total = torch.cuda.mem_get_info()
    return total - free


def _to_gb(size: int):
    return size / 1024 ** 3


def _record_event():
    event = torch.cuda.Event(enable_timing=True)
    event.record()
    return event


def _world_20_80_percentile_avg(nums: List[float]):
    nums.sort()
    start = int(len(nums) * 0.2)
    end = int(len(nums) * 0.8)
    avg = sum(nums[start:end]) / (end - start)
    tensor = torch.tensor(avg).cuda()
    dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
    return tensor.item()


def system_auto_config():
    # 0. Measure VRAM and DRAM usage
    curr_gpu = torch.cuda.current_device()
    curr_gpu = torch.cuda.get_device_properties(curr_gpu)
    total_vram = curr_gpu.total_memory * ACCESSIBLE_GPU_MEMORY_RATIO
    baseline_vram = _save_vram_usage_estimation()
    usable_vram = total_vram - baseline_vram

    dist.rank0_logger.info(
        "\n\n> FlexTrain system GPU memory usage estimation:\n"
        f"  - Estimated total VRAM size: {_to_gb(total_vram):.3f} GB\n"
        f"  - Estimated baseline memory: {_to_gb(baseline_vram):.3f} GB\n"
        f"  - Estimated usable memory: {_to_gb(usable_vram):.3f} GB\n"
    )

    process_used_dram = psutil.Process().memory_info().rss
    process_baseline_dram = process_used_dram - get_allocated_dram_size()
    process_baseline_dram = torch.tensor(process_baseline_dram).cuda()
    dist.all_reduce(process_baseline_dram, op=dist.ReduceOp.SUM)
    baseline_dram = process_baseline_dram.item()

    total_dram = get_flextrain_config().max_dram_usage * 1024 ** 3
    usable_dram = total_dram - baseline_dram
    usable_dram = usable_dram * ACCESSIBLE_CPU_MEMORY_RATIO

    dist.rank0_logger.info(
        "\n\n> FlexTrain system CPU memory usage estimation:\n"
        f"  - Estimated total DRAM size: {_to_gb(total_dram):.3f} GB\n"
        f"  - Estimated baseline memory: {_to_gb(baseline_dram):.3f} GB\n"
        f"  - Estimated usable memory: {usable_dram / 1024 ** 3:.3f} GB\n"
    )

    # 1. Benchmark the performance of the target LLM
    pre_fwd_times, pre_bwd_times, post_times = [], [], []
    layer_fwd_times, layer_bwd_times = [], []

    # Get the first LLM layer ready
    get_para_coordinator()._link_unit_parameters(
        0, get_para_coordinator()._gpu_available_paras
    )
    get_opts_coordinator().pre_unit_backward(0)

    iterations = tqdm(
        range(BENCHMARK_GPU_ITERATIONS), desc="LLM Perf. Benchmarking"
    ) if dist.get_rank() == 0 else range(BENCHMARK_GPU_ITERATIONS)
    for _ in iterations:
        # 1. Benchmark the forward time of pre-process
        pre_fwd_start = _record_event()
        pre_inputs, post_inputs, loss_inputs = LLMFunc.get_batch()
        first_passed_down, every_layer = LLMFunc.pre_process(pre_inputs)
        pre_fwd_end = _record_event()

        # 2. Benchmark the forward time of each layer
        passed_down = detach_variable(first_passed_down)
        unit_fwd_start = _record_event()
        unit_passed_down = LLMFunc.layer_forward(0, 1)(
            *passed_down, *every_layer
        )
        unit_fwd_end = _record_event()

        # IMPORTANT: initialize the interlayer coordinator
        passed_down_tensor = retrieve_tensor(unit_passed_down)
        get_interlayer_coordinator()._init_coordinator(passed_down_tensor)

        # 3. Benchmark the post-process time
        last_passed_down = detach_variable(unit_passed_down)
        post_start = _record_event()
        llm_outputs = LLMFunc.post_process(last_passed_down, post_inputs)
        llm_loss_rst = LLMFunc.loss(llm_outputs, loss_inputs)
        loss = retrieve_llm_loss(llm_loss_rst)
        torch.autograd.backward(loss)
        post_end = _record_event()

        # 4. Benchmark the backward time of each layer
        tensor = retrieve_tensor(unit_passed_down)
        gradient = torch.ones_like(tensor)
        unit_bwd_start = _record_event()
        torch.autograd.backward(tensor, gradient)
        unit_bwd_end = _record_event()

        # 5. Benchmark the backward time of pre-process
        tensor = retrieve_tensor(first_passed_down)
        gradient = torch.ones_like(tensor)
        pre_bwd_start = _record_event()
        torch.autograd.backward(tensor, gradient)
        pre_bwd_end = _record_event()

        # Record the time
        torch.cuda.synchronize()
        pre_fwd_times.append(pre_fwd_start.elapsed_time(pre_fwd_end))
        pre_bwd_times.append(pre_bwd_start.elapsed_time(pre_bwd_end))
        post_times.append(post_start.elapsed_time(post_end))
        layer_fwd_times.append(unit_fwd_start.elapsed_time(unit_fwd_end))
        layer_bwd_times.append(unit_bwd_start.elapsed_time(unit_bwd_end))

    # Compute the average time
    avg_pre_fwd = _world_20_80_percentile_avg(pre_fwd_times)
    avg_pre_bwd = _world_20_80_percentile_avg(pre_bwd_times)
    avg_post = _world_20_80_percentile_avg(post_times)
    avg_layer_fwd = _world_20_80_percentile_avg(layer_fwd_times)
    avg_layer_bwd = _world_20_80_percentile_avg(layer_bwd_times)
    avg_pre_post = avg_pre_fwd + avg_pre_bwd + avg_post

    dist.rank0_logger.info(
        "\n\n> FlexTrain model performance benchmarking results"
        f" (micro_batch_size={get_flextrain_config().micro_batch_size}):\n"
        f"  - Average pre-process forward time: {avg_pre_fwd:.3f} ms\n"
        f"  - Average pre-process backward time: {avg_pre_bwd:.3f} ms\n"
        f"  - Average post-process time: {avg_post:.3f} ms\n"
        f"  - Average layer forward time: {avg_layer_fwd:.3f} ms\n"
        f"  - Average layer backward time: {avg_layer_bwd:.3f} ms\n"
    )

    # 2. Benchmark the PCIe bandwidth
    pcie_times = []
    read_buffer = torch.empty(
        BENCHMARK_PCIE_BLOCK_SIZE, dtype=torch.uint8, pin_memory=True
    )
    write_buffer = torch.empty(
        BENCHMARK_PCIE_BLOCK_SIZE, dtype=torch.uint8,
        device=torch.cuda.current_device()
    )
    iterations = tqdm(
        range(BENCHMARK_PCIE_ITERATIONS), desc="PCIe Benchmarking"
    ) if dist.get_rank() == 0 else range(BENCHMARK_PCIE_ITERATIONS)
    for _ in iterations:
        write_buffer += 1
        pcie_start = _record_event()
        read_buffer.copy_(write_buffer)
        pcie_end = _record_event()
        # Record the time
        torch.cuda.synchronize()
        pcie_times.append(pcie_start.elapsed_time(pcie_end))

    # Compute the average time
    avg_pcie = _world_20_80_percentile_avg(pcie_times) / 1000
    pcie_bandwidth = _to_gb(BENCHMARK_PCIE_BLOCK_SIZE) / avg_pcie
    dist.rank0_logger.info(
        "\n\n> FlexTrain PCIe bandwidth benchmarking results:\n"
        f"  - Average PCIe bandwidth: {pcie_bandwidth:.3f} GB/s\n"
    )
    pcie_bandwidth = BENCHMARK_PCIE_BLOCK_SIZE / avg_pcie

    # 3. Benchmark the NVMe swapper
    swap_times = []
    swapper = get_nvme_swapper()
    swapper.synchronize()

    read_buffer = torch.empty(BENCHMARK_NVME_BLOCK_SIZE, dtype=torch.uint8)
    write_buffer = torch.empty(BENCHMARK_NVME_BLOCK_SIZE, dtype=torch.uint8)

    for i in range(BENCHMARK_NUM_BLOCKS):
        write_buffer += 1
        swapper.swap_out(
            FlexTrainDataID(FlexTrainDataType.PARA, i), write_buffer
        )

    # Benchmark the overlap swap
    iterations = tqdm(
        range(BENCHMARK_NVME_ITERATIONS), desc="NVMe Read-Write Benchmarking"
    ) if dist.get_rank() == 0 else range(BENCHMARK_NVME_ITERATIONS)
    for i in iterations:
        read_index = (i + 1) % BENCHMARK_NUM_BLOCKS
        write_index = i % BENCHMARK_NUM_BLOCKS
        overlap_start = time.time()
        swapper.swap_in(
            FlexTrainDataID(FlexTrainDataType.PARA, read_index),
            read_buffer, async_op=True
        )
        swapper.swap_out(
            FlexTrainDataID(FlexTrainDataType.PARA, write_index),
            write_buffer, async_op=True
        )
        swapper.synchronize()
        overlap_end = time.time()
        swap_times.append(overlap_end - overlap_start)

    # Compute the average time
    avg_swap_time = _world_20_80_percentile_avg(swap_times)
    # Compute the bandwidth
    avg_swap_bandwidth = 2 * BENCHMARK_NVME_BLOCK_SIZE / avg_swap_time
    nvme_swap = _to_gb(avg_swap_bandwidth)
    world_size = dist.get_world_size()
    dist.rank0_logger.info(
        "\n\n> FlexTrain NVMe swapper benchmarking results:\n"
        f"  - Average swap bandwidth: {world_size} x {nvme_swap:.3f} GB/s\n"
    )

    # 4. Benchmark the optimizer step throughput
    cpu_optimizer = get_opts_coordinator()._optimizer.cpu_optimizer
    # Initialize the optimizer states
    cpu_optimizer.profile_step(
        BENCHMARK_OPTIMIZER_BLOCK_SIZE,
        dtype=get_flextrain_config().mixed_precision.master_dtype
    )
    # TEMPORARY: use opt_state_per_element to estimate the memory usage
    opt_state_per_element = sum(
        p.numel() for p in cpu_optimizer._init_optimizer_states(
            1, get_flextrain_config().mixed_precision.master_dtype
        )
    )
    step_times = []
    iterations = tqdm(
        range(BENCHMARK_CPU_ITERATIONS), desc="Optimizer Step Benchmarking"
    ) if dist.get_rank() == 0 else range(BENCHMARK_CPU_ITERATIONS)
    for _ in iterations:
        opt_step_start = time.time()
        cpu_optimizer.profile_step(
            BENCHMARK_OPTIMIZER_BLOCK_SIZE,
            dtype=get_flextrain_config().mixed_precision.master_dtype
        )
        opt_step_end = time.time()
        step_times.append(opt_step_end - opt_step_start)

    # Compute the average time
    avg_opt_step = _world_20_80_percentile_avg(step_times)
    step_throughput = _to_gb(BENCHMARK_OPTIMIZER_BLOCK_SIZE) / avg_opt_step
    dist.rank0_logger.info(
        "\n\n> FlexTrain optimizer step throughput benchmarking results:\n"
        f"  - Average optimizer step throughput: {step_throughput:.3f} G/s\n"
    )
    step_throughput = BENCHMARK_OPTIMIZER_BLOCK_SIZE / avg_opt_step

    # 5. Fill in the machine performance parameters
    mixed_precision = get_flextrain_config().mixed_precision
    perf_params = MachinePerfParams(
        num_layers=get_para_coordinator().num_layers,
        opt_state_per_element=opt_state_per_element,
        device_dtype_itemsize=mixed_precision.device_dtype.itemsize,
        gradacc_dtype_itemsize=mixed_precision.gradacc_dtype.itemsize,
        master_dtype_itemsize=mixed_precision.master_dtype.itemsize,
        pre_post_process_time=avg_pre_post / 1000,
        layer_fwd_time=avg_layer_fwd / 1000,
        layer_bwd_time=avg_layer_bwd / 1000,
        usable_vram=usable_vram,
        usable_dram=usable_dram,
        layer_numel=get_para_coordinator().unit_numel,
        interlayer_numel=get_interlayer_coordinator()._tensor_numel,
        pcie_bandwidth=pcie_bandwidth,
        nvme_swap_bandwidth=avg_swap_bandwidth * world_size,
        opt_step_throughput=step_throughput
    )

    # 6. Solve the optimization problem
    solver = FlexTrainConfigSolver(perf_params)
    solver.solve()

    # Exit the script
    exit()


@dataclass
class MachinePerfParams:
    # Basic parameters
    num_layers: int
    opt_state_per_element: int
    device_dtype_itemsize: int
    gradacc_dtype_itemsize: int
    master_dtype_itemsize: int

    # Time in seconds
    pre_post_process_time: float
    layer_fwd_time: float
    layer_bwd_time: float

    # Memory in bytes
    usable_vram: int
    usable_dram: int

    # Size in number of elements
    layer_numel: int
    interlayer_numel: int

    # Bandwidth in B/s
    pcie_bandwidth: float
    nvme_swap_bandwidth: float

    # Throughput in element/s
    opt_step_throughput: float


@dataclass
class ConfigResult:
    iteration_time: float
    mbpr: int
    alpha: float
    cpu_ckpt: float
    cpu_grad: float
    cpu_para: float
    cpu_opts: float


class FlexTrainConfigSolver:

    def __init__(self, machine_perf_params: MachinePerfParams):
        # Disable the logging
        pulp.LpSolverDefault.msg = False

        # Get basic parameters
        self.world_size = dist.get_world_size()
        # Link to the machine performance parameters
        self.mpp = machine_perf_params

        # Define the decision variables
        # Interlayer checkpoint
        # There is no GPU checkpoint
        self.cpu_ckpt = LpVariable("cpu_ckpt", 0, 1)
        self.nvme_ckpt = LpVariable("nvme_ckpt", 0, 1)

        # Interlayer gradient
        self.gpu_grad = LpVariable("gpu_grad", 0, 1)
        self.cpu_grad = LpVariable("cpu_grad", 0, 1)
        # There is no NVMe gradient

        # Parameters
        # There is no GPU parameter
        self.cpu_para = LpVariable("cpu_para", 0, 1)
        self.nvme_para = LpVariable("nvme_para", 0, 1)

        # Optimizer state
        # There is no GPU optimizer state
        self.cpu_opts = LpVariable("cpu_opts", 0, 1)
        self.nvme_opts = LpVariable("nvme_opts", 0, 1)

        # Auxiliary variables
        self.unit_fwd_time = LpVariable("unit_fwd_time", 0)
        self.unit_bwd_time = LpVariable("unit_bwd_time", 0)

    def solve(self):
        mbpr = 0
        max_throughput = 0
        best_results: List[ConfigResult] = []
        extra_configs = NUM_EXTRA_CONFIGS
        while True:
            mbpr += 1
            mbpr_max_throughput = 0
            mbpr_best_result = None

            # Try all potential alpha values
            for alpha in POTENTIAL_ALPHA_VALUES:
                result = self._solve_given_hypers(mbpr, alpha)
                if result is None:
                    continue
                new_throughput = result.mbpr / result.iteration_time
                if new_throughput >= \
                        mbpr_max_throughput * THROUGHPUT_STABLE_THRESHOLD:
                    mbpr_max_throughput = new_throughput
                    mbpr_best_result = result

            best_results.append(mbpr_best_result)
            if mbpr_best_result is None:
                continue

            # Update the max throughput if it is significantly better
            if mbpr_max_throughput >= \
                    max_throughput * THROUGHPUT_STABLE_THRESHOLD:
                max_throughput = mbpr_max_throughput
            else:
                if extra_configs == 0:
                    break
                else:
                    extra_configs -= 1

        for result in best_results:
            if result is None:
                continue
            mbps = result.mbpr / result.iteration_time
            dist.print_rank0(
                f"configuration {result} throughput: {mbps:.3f} micro-batch/s"
            )

    def _solve_given_hypers(self, mbpr: int, alpha: float):
        # Create a new problem instance
        self.problem = LpProblem(f"AutoConfig_mbpr{mbpr}_alpha{alpha}")
        self.mbpr = mbpr
        self.alpha = alpha

        # Define the objective function
        iteration_time = self.unit_fwd_time + self.unit_bwd_time
        self.problem += \
            iteration_time + \
            REGULARIZATION_PENALTY * (self.nvme_para + self.nvme_opts), \
            "Objective function"

        # Define the constraints
        # 0. Basic constraints
        self._add_basic_constraints()

        # 1. Memory constraints
        self._add_mem_constraint()

        # 2. Computation constraints
        self._add_computation_constraint()

        # 3. Bandwidth constraints
        self._add_bandwidth_constraint()

        # Solve the problem
        status = self.problem.solve()

        if status == 1:
            iteration_time = self.unit_fwd_time + self.unit_bwd_time
            iteration_time = pulp.value(iteration_time)
            iteration_time = iteration_time * self.mpp.num_layers
            iteration_time += self.mpp.pre_post_process_time * mbpr
            return ConfigResult(
                iteration_time=iteration_time,
                mbpr=mbpr,
                alpha=alpha,
                cpu_ckpt=pulp.value(self.cpu_ckpt),
                cpu_grad=pulp.value(self.cpu_grad),
                cpu_para=pulp.value(self.cpu_para),
                cpu_opts=pulp.value(self.cpu_opts)
            )
        else:
            return None

    def _add_basic_constraints(self):
        # 1. Interlayer checkpoint
        self.problem += self.cpu_ckpt + self.nvme_ckpt == 1, \
            "Interlayer checkpoint sum constraint"

        # 2. Interlayer gradient
        self.problem += self.gpu_grad + self.cpu_grad == 1, \
            "Interlayer gradient sum constraint"

        # 3. Parameters
        self.problem += self.cpu_para + self.nvme_para == 1, \
            "Parameters sum constraint"

        # 4. Optimizer state
        self.problem += self.cpu_opts + self.nvme_opts == 1, \
            "Optimizer state sum constraint"

    def _add_mem_constraint(self):
        # Basic parameters
        mbpr = self.mbpr
        alpha = self.alpha

        # Figure out the data types and their sizes
        device_element_size = self.mpp.device_dtype_itemsize
        gradacc_element_size = self.mpp.gradacc_dtype_itemsize
        master_element_size = self.mpp.master_dtype_itemsize

        # Compute base buffer sizes
        rank_ckpt_size = mbpr * self.mpp.interlayer_numel * self.mpp.num_layers
        rank_ckpt_size *= device_element_size
        world_ckpt_size = rank_ckpt_size * self.world_size

        rank_grad_size = mbpr * self.mpp.interlayer_numel
        rank_grad_size *= device_element_size

        world_para_size = self.mpp.layer_numel * self.mpp.num_layers
        world_para_size *= device_element_size

        world_fwd_grad_size = self.mpp.layer_numel * self.mpp.num_layers
        world_fwd_grad_size *= gradacc_element_size * alpha

        world_opts_size = self.mpp.layer_numel * self.mpp.num_layers
        world_opts_size *= master_element_size * \
            (self.mpp.opt_state_per_element + 1)

        # Add the GPU memory constraint
        self.problem += self.gpu_grad * rank_grad_size \
            <= self.mpp.usable_vram, "GPU memory constraint"

        # Figure out CPU memory consumptions on each buffer
        rank_cpu_ckpt_base = self.cpu_ckpt * rank_ckpt_size
        rank_cpu_grad_base = self.cpu_grad * rank_grad_size
        rank_nvme_ckpt_margin_base = \
            self.nvme_ckpt * self.mpp.interlayer_numel * min(mbpr, 4) * \
            self.mpp.num_layers * device_element_size
        rank_nvme_ckpt_prefetch_buffer = \
            self.nvme_ckpt * self.mpp.interlayer_numel * max(mbpr - 4, 0) * \
            2 * device_element_size

        world_cpu_ckpt_base = self.cpu_ckpt * world_ckpt_size
        world_cpu_para_base = self.cpu_para * world_para_size
        world_nvme_para_prefetch_buffer = \
            self.nvme_para * self.mpp.layer_numel * device_element_size

        world_cpu_opts_base = self.cpu_opts * world_opts_size
        world_cpu_opt_grad_buffer = \
            max(alpha, 1 - alpha) * self.mpp.layer_numel * 2 * \
            master_element_size
        world_cpu_opt_work_buffer = \
            max(alpha, 1 - alpha) * self.mpp.layer_numel * 3 * \
            (self.mpp.opt_state_per_element * self.nvme_opts + 1) * \
            master_element_size
        world_nvme_para_update_buffer = \
            self.nvme_para * (1 - alpha) * self.mpp.layer_numel * 2 * \
            device_element_size

        # Add the CPU memory constraint
        self.problem += \
            dist.get_world_size() * (
                rank_cpu_ckpt_base + rank_cpu_grad_base +
                rank_nvme_ckpt_margin_base + rank_nvme_ckpt_prefetch_buffer
            ) + world_cpu_para_base + world_nvme_para_prefetch_buffer + \
            world_cpu_opts_base + world_cpu_opt_grad_buffer + \
            world_cpu_opt_work_buffer + world_nvme_para_update_buffer <= \
            self.mpp.usable_dram, \
            "CPU memory constraint"

        # Add alpha constraint
        self.problem += world_cpu_ckpt_base >= world_fwd_grad_size, \
            "Alpha lower bound constraint"

    def _add_computation_constraint(self):
        # Basic parameters
        mbpr = self.mbpr
        alpha = self.alpha

        # 1. Forward / Backward time
        self.problem += self.unit_fwd_time >= self.mpp.layer_fwd_time * mbpr, \
            "Forward layer computation constraint"
        # Attention: add extra recomputation time
        self.problem += self.unit_bwd_time >= \
            (
                self.mpp.layer_fwd_time + self.mpp.layer_bwd_time
            ) * mbpr, "Backward layer computation constraint"

        # 2. Optimizer step throughput
        rank_step_numel = self.mpp.layer_numel / self.world_size
        self.problem += self.unit_fwd_time * self.mpp.opt_step_throughput >= \
            alpha * rank_step_numel, "Forward optimizer step constraint"
        self.problem += self.unit_bwd_time * self.mpp.opt_step_throughput >= \
            (1 - alpha) * rank_step_numel, "Backward optimizer step constraint"

    def _add_bandwidth_constraint(self):
        # Basic parameters
        mbpr = self.mbpr
        alpha = self.alpha

        # Figure out dtype itemsize
        device_element_size = self.mpp.device_dtype_itemsize
        gradacc_element_size = self.mpp.gradacc_dtype_itemsize
        master_element_size = self.mpp.master_dtype_itemsize

        # Figure out layer-level data sizes
        rank_ckpt = self.mpp.interlayer_numel * mbpr * device_element_size
        rank_act_grad = self.mpp.interlayer_numel * mbpr * device_element_size
        rank_para = self.mpp.layer_numel * device_element_size
        rank_acc_grad = self.mpp.layer_numel * gradacc_element_size

        # 1. PCIe bandwidth (due to FlexTrain's design, we add each operation)
        #    Attention: each process does not share the same PCIe bandwidth
        self.problem += rank_ckpt * 2 + rank_para <= \
            self.mpp.pcie_bandwidth * self.unit_fwd_time, \
            "Forward PCIe bandwidth constraint"

        self.problem += rank_ckpt + rank_act_grad * 2 + \
            rank_para + rank_acc_grad <= \
            self.mpp.pcie_bandwidth * self.unit_bwd_time, \
            "Backward PCIe bandwidth constraint"

        # Figure out the layer-level NVMe data sizes
        rank_nvme_ckpt = self.nvme_ckpt * self.mpp.interlayer_numel * \
            max(mbpr - 4, 0) * device_element_size
        world_nvme_para = self.nvme_para * self.mpp.layer_numel * \
            device_element_size
        world_nvme_opts = self.nvme_opts * self.mpp.layer_numel * \
            master_element_size * (self.mpp.opt_state_per_element + 1)

        # 2. NVMe bandwidth (due to SSD's design, we add each operation)
        #    Attention: each process DOES share the same NVMe bandwidth
        self.problem += \
            (
                (1 - alpha) * world_nvme_para +
                alpha * world_nvme_opts
            ) / self.mpp.nvme_swap_bandwidth + \
            (
                rank_nvme_ckpt * self.world_size +
                alpha * world_nvme_para +
                alpha * world_nvme_opts
            ) / self.mpp.nvme_swap_bandwidth <= self.unit_fwd_time, \
            "Forward NVMe bandwidth constraint"

        self.problem += \
            (
                rank_nvme_ckpt * self.world_size +
                world_nvme_para +
                (1 - alpha) * world_nvme_opts
            ) / self.mpp.nvme_swap_bandwidth + \
            (
                (1 - alpha) * world_nvme_para +
                (1 - alpha) * world_nvme_opts
            ) / self.mpp.nvme_swap_bandwidth <= self.unit_bwd_time, \
            "Backward NVMe bandwidth constraint"
