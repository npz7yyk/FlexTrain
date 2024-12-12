import time
import torch

from dataclasses import dataclass
from pulp import LpProblem, LpVariable, LpSolverDefault
from tqdm import tqdm
from typing import List

from flextrain.checkpointing import detach_variable, retrieve_tensor
from flextrain.config import get_flextrain_config
from flextrain.defaults import (
    BENCHMARK_GPU_ITERATIONS,
    BENCHMARK_PCIE_ITERATIONS,
    BENCHMARK_NVME_ITERATIONS,
    BENCHMARK_CPU_ITERATIONS,
    BENCHMARK_NUM_BLOCKS,
    BENCHMARK_BLOCK_SIZE,
    ACCESSIBLE_GPU_MEMORY_RATIO,
    ACCESSIBLE_CPU_MEMORY_RATIO
)
from flextrain.llm_func import LLMFunc, retrieve_llm_loss
from flextrain.memory import FlexTrainDataType
from flextrain.memory.coordinator import (
    get_para_coordinator, get_interlayer_coordinator, get_opts_coordinator
)
from flextrain.memory.nvme_swapper import get_nvme_swapper, FlexTrainDataID
from flextrain.utils import dist


def _save_memory_estimation():
    """ Estimate the memory usage. Return the total memory in bytes. """
    free, total = torch.cuda.mem_get_info()
    return total - free


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
    # 1. Benchmark the performance of the target LLM
    pre_fwd_times, pre_bwd_times, post_times = [], [], []
    layer_fwd_times, layer_bwd_times = [], []
    max_peak_mem = 0

    # Get the first LLM layer ready
    get_para_coordinator()._link_unit_parameters(
        0, get_para_coordinator()._gpu_available_paras
    )
    get_opts_coordinator().pre_unit_backward(0)

    iterations = tqdm(
        range(BENCHMARK_GPU_ITERATIONS), desc="LLM Perf. Benchmarking"
    ) if dist.get_rank() == 0 else range(BENCHMARK_GPU_ITERATIONS)
    for _ in iterations:
        # Peak memory allocated before running layer forward/backward
        peak_mem = _save_memory_estimation()
        if peak_mem > max_peak_mem:
            max_peak_mem = peak_mem

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

    dist.rank0_logger.info(
        "\n\n> FlexTrain model performance benchmarking results"
        f" (micro_batch_size={get_flextrain_config().micro_batch_size}):\n"
        f"  - Average pre-process forward time: {avg_pre_fwd:.3f} ms\n"
        f"  - Average pre-process backward time: {avg_pre_bwd:.3f} ms\n"
        f"  - Average post-process time: {avg_post:.3f} ms\n"
        f"  - Average layer forward time: {avg_layer_fwd:.3f} ms\n"
        f"  - Average layer backward time: {avg_layer_bwd:.3f} ms\n"
        f"  - Peak GPU memory usage: {max_peak_mem / 1024 ** 3:.3f} GB\n"
    )

    # 2. Benchmark the PCIe bandwidth
    pcie_times = []
    read_buffer = torch.empty(
        BENCHMARK_BLOCK_SIZE, dtype=torch.uint8, pin_memory=True
    )
    write_buffer = torch.empty(
        BENCHMARK_BLOCK_SIZE, dtype=torch.uint8,
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
    pcie_bandwidth = BENCHMARK_BLOCK_SIZE / 1024 ** 3 / avg_pcie

    dist.rank0_logger.info(
        "\n\n> FlexTrain PCIe bandwidth benchmarking results:\n"
        f"  - Average PCIe bandwidth: {pcie_bandwidth:.3f} GB/s\n"
    )

    # 3. Benchmark the NVMe swapper
    swap_in_times, swap_out_times, overlap_times = [], [], []
    swapper = get_nvme_swapper()

    read_buffer = torch.empty(BENCHMARK_BLOCK_SIZE, dtype=torch.uint8)
    write_buffer = torch.empty(BENCHMARK_BLOCK_SIZE, dtype=torch.uint8)

    for i in range(BENCHMARK_NUM_BLOCKS):
        write_buffer += 1
        swapper.swap_out(
            FlexTrainDataID(FlexTrainDataType.PARA, i), write_buffer
        )

    # Benchmark the swap-in time
    iterations = tqdm(
        range(BENCHMARK_NVME_ITERATIONS), desc="NVMe Read Benchmarking"
    ) if dist.get_rank() == 0 else range(BENCHMARK_NVME_ITERATIONS)
    for i in iterations:
        read_index = i % BENCHMARK_NUM_BLOCKS
        swap_in_start = time.time()
        swapper.swap_in(
            FlexTrainDataID(FlexTrainDataType.PARA, read_index), read_buffer
        )
        swap_in_end = time.time()
        swap_in_times.append(swap_in_end - swap_in_start)

    # Benchmark the swap-out time
    iterations = tqdm(
        range(BENCHMARK_NVME_ITERATIONS), desc="NVMe Write Benchmarking"
    ) if dist.get_rank() == 0 else range(BENCHMARK_NVME_ITERATIONS)
    for i in iterations:
        write_index = i % BENCHMARK_NUM_BLOCKS
        swap_out_start = time.time()
        swapper.swap_out(
            FlexTrainDataID(FlexTrainDataType.PARA, write_index), write_buffer
        )
        swap_out_end = time.time()
        swap_out_times.append(swap_out_end - swap_out_start)

    # Benchmark the overlap time
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
        overlap_times.append(overlap_end - overlap_start)

    # Compute the average time
    avg_swap_in = _world_20_80_percentile_avg(swap_in_times)
    avg_swap_out = _world_20_80_percentile_avg(swap_out_times)
    avg_overlap = _world_20_80_percentile_avg(overlap_times)
    # Compute the bandwidth
    swap_in = BENCHMARK_BLOCK_SIZE / 1024 ** 3 / avg_swap_in
    swap_out = BENCHMARK_BLOCK_SIZE / 1024 ** 3 / avg_swap_out
    overlap = BENCHMARK_BLOCK_SIZE / 1024 ** 3 / avg_overlap

    world_size = dist.get_world_size()
    dist.rank0_logger.info(
        "\n\n> FlexTrain NVMe swapper benchmarking results:\n"
        f"  - Average swap-in bandwidth: {world_size} x {swap_in:.3f} GB/s\n"
        f"  - Average swap-out bandwidth: {world_size} x {swap_out:.3f} GB/s\n"
        f"  - Average overlap bandwidth: {world_size} x {overlap:.3f} GB/s\n"
    )

    # 4 Benchmark the optimizer step throughput
    cpu_optimizer = get_opts_coordinator()._optimizer
    from flextrain.ops.adam.cpu_adam import FlexTrainCPUAdam
    assert isinstance(cpu_optimizer, FlexTrainCPUAdam), \
        f"Only FlexTrainAdam is supported for now, got {type(cpu_optimizer)}"
    step_times = []
    iterations = tqdm(
        range(BENCHMARK_CPU_ITERATIONS), desc="Optimizer Step Benchmarking"
    ) if dist.get_rank() == 0 else range(BENCHMARK_CPU_ITERATIONS)
    for _ in iterations:
        opt_step_start = time.time()
        cpu_optimizer.profile_step(
            BENCHMARK_BLOCK_SIZE,
            dtype=get_flextrain_config().mixed_precision.master_dtype
        )
        opt_step_end = time.time()
        step_times.append(opt_step_end - opt_step_start)

    # Compute the average time
    avg_opt_step = _world_20_80_percentile_avg(step_times)
    step_throughput = BENCHMARK_BLOCK_SIZE / 1024 ** 3 / avg_opt_step

    dist.rank0_logger.info(
        "\n\n> FlexTrain optimizer step throughput benchmarking results:\n"
        f"  - Average optimizer step throughput: {step_throughput:.3f} G/s\n"
    )

    # Exit the script
    exit()


@dataclass
class MachinePerfParams:
    # Basic parameters
    num_layers: int
    opt_state_per_element: int

    # Time in milliseconds
    pre_process_fwd_time: float
    pre_process_bwd_time: float
    post_process_time: float
    layer_fwd_time: float
    layer_bwd_time: float

    # Memory in bytes
    peak_act_mem: int

    # Size in number of elements
    layer_numel: int
    nonlayer_numel: int
    interlayer_numel: int

    # Bandwidth in GB/s
    pcie_bandwidth: float
    nvme_r_bandwidth: float
    nvme_w_bandwidth: float
    nvme_rw_bandwidth: float
    opt_step_throughput: float


class FlexTrainConfigSolver:

    def __init__(self, machine_perf_params: MachinePerfParams):
        # Disable the logging
        LpSolverDefault.msg = False

        # Link to the machine performance parameters
        self.mpp = machine_perf_params

        # Define the decision variables
        # Interlayer checkpoint
        self.gpu_ckpt = LpVariable("gpu_ckpt", 0, 1)
        self.cpu_ckpt = LpVariable("cpu_ckpt", 0, 1)

        # Interlayer gradient
        self.gpu_grad = LpVariable("gpu_grad", 0, 1)
        self.cpu_grad = LpVariable("cpu_grad", 0, 1)

        # Parameters
        self.gpu_para = LpVariable("gpu_para", 0, 1)
        self.cpu_para = LpVariable("cpu_para", 0, 1)

        # Optimizer state
        # There is no GPU optimizer state
        self.cpu_opts = LpVariable("cpu_opts", 0, 1)

        # Alpha split
        self.alpha = LpVariable("alpha", 0, 0.5)

        # Auxiliary variables
        self.unit_fwd_time = LpVariable("unit_fwd_time", 0)
        self.unit_bwd_time = LpVariable("unit_bwd_time", 0)

    def _solve_given_mbpr(self, mbpr: int):
        # Create the problem instance
        problem = LpProblem()

        # Define the objective function
        num_layers = self.mpp.num_layers
        problem += \
            self.unit_fwd_time * (num_layers - 1) + \
            self.unit_bwd_time * num_layers, "Objective"

        # Define the constraints
        # 1. Memory constraints
        self._add_gpu_mem_constraint(problem, mbpr)

        # 2. Bandwidth constraints
        # self._add_bandwidth_constraint(problem)

        # Solve the problem
        status = problem.solve()

        # Display the results
        if status == 1:
            dist.rank0_logger.info(
                f"Optimal value: {problem.objective.value()}"
            )
            dist.rank0_logger.info(
                f"Optimal solution: {self.gpu_ckpt.value()} "
                f"{self.cpu_ckpt.value()} {self.gpu_grad.value()} "
                f"{self.cpu_grad.value()} {self.gpu_para.value()} "
                f"{self.cpu_para.value()} {self.cpu_opts.value()} "
                f"{self.alpha.value()}"
            )
        else:
            dist.rank0_logger.info(f"Optimization failed: {problem.status}")

    def _add_gpu_mem_constraint(self, problem: LpProblem, mbpr: int):
        # Figure out the data types and their sizes
        mixed_precision = get_flextrain_config().mixed_precision
        device_dtype = mixed_precision.device_dtype
        gradacc_dtype = mixed_precision.gradacc_dtype
        device_element_size = device_dtype.itemsize
        gradacc_element_size = gradacc_dtype.itemsize

        # 0. Get the total GPU memory
        curr_gpu = torch.cuda.current_device()
        curr_gpu = torch.cuda.get_device_properties(curr_gpu)
        total_gpu_mem = curr_gpu.total_memory
        total_gpu_mem = total_gpu_mem * ACCESSIBLE_GPU_MEMORY_RATIO
