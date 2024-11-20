import torch

from abc import ABC, abstractmethod
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from torch import Tensor
from tqdm import tqdm
from typing import Set, Dict

from flextrain.checkpointing import set_post_recomputation_function
from flextrain.config import get_flextrain_config
from flextrain.memory import (
    Waitable,
    DummyHandle,
    FunctionHandle,
    FlexTrainDataType as Dtype,
    FlexTrainDataID,
    RotateContainer,
    get_split_numels,
    allocate_memory_chunks,
    copy_segments,
    get_data_stream
)
from flextrain.memory.coordinator import (
    get_para_coordinator,
    get_interlayer_coordinator
)
from flextrain.memory.nvme_swapper import NVMeGroup
from flextrain.scheduler import LLMTask
from flextrain.utils import dist
from flextrain.utils.logging import rank0_logger


@dataclass
class OptTarget(ABC):
    para: Tensor
    grad: Tensor


STEP_KEY = "step"


class FlexTrainCPUOptimizer(ABC):
    FWD_INDEX = 0
    BWD_INDEX = 1

    def __init__(self, unit_group_map: Dict[int, Dict]):
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._unit_group_map = unit_group_map

        # Initialize the state for each unit.
        if not hasattr(self, "state"):
            self.state = defaultdict(dict)
        for unit in unit_group_map:
            self.state[unit] = {STEP_KEY: 0}

    @abstractmethod
    def _step(self, step: int, args: Dict, opt_target: OptTarget):
        pass

    def update_state(self):
        for unit in self._unit_group_map:
            self.state[unit][STEP_KEY] += 1

    def _submit_step(self, unit_index: int, opt_target: OptTarget):
        assert unit_index in self._unit_group_map, (
            "The unit index is not in the unit group map."
        )

        # Submit the step function to the executor.
        future = self._executor.submit(
            self._step,
            self.state[unit_index][STEP_KEY],
            self._unit_group_map[unit_index],
            opt_target
        )
        return FunctionHandle(future.result)

    @abstractmethod
    def submit_step(
        self, unit_index: int, para: torch.Tensor, grad: torch.Tensor,
        *args, **kwargs
    ) -> Waitable:
        pass


def _convert_dtype_view(
    tensor: Tensor,
    target_dtype: torch.dtype
) -> Tensor:
    if tensor.numel() == 0:
        assert tensor.shape[-1] == 0
        return torch.empty_like(tensor, dtype=target_dtype)
    return tensor.view(target_dtype)


@dataclass
class MicroBatchTask(LLMTask):
    forward: bool


class FlexTrainOptsCoordinator:
    TRANSFER_OPTS = 0
    RECOVER_GRAD = 0
    TRANSFER_GRAD = 0
    OPTIMIZER_STEP = 1
    OFFLOAD_GRAD = 1
    UPDATE_OPTS = 2
    UPDATE_PARA = 2
    OFFLOAD_PARA = 3

    def __init__(self):
        # Lazy initialization of optimizer coordinator.
        self._initialized = False

    @property
    def is_initialized(self):
        return self._initialized

    @property
    def _gpu_bwd_receive_grads(self):
        return self._gpu_bwd_grad_buffers[0]

    @property
    def _gpu_bwd_transfer_grads(self):
        return self._gpu_bwd_grad_buffers[1]

    @property
    def _cpu_opt_receive_grads(self):
        return self._cpu_opt_grad_buffers[0]

    @property
    def _cpu_opt_available_grads(self):
        return self._cpu_opt_grad_buffers[1]

    @property
    def _cpu_opt_receive_states(self):
        return self._cpu_opt_work_buffers[0]

    @property
    def _cpu_opt_available_states(self):
        return self._cpu_opt_work_buffers[1]

    @property
    def _cpu_opt_transfer_states(self):
        return self._cpu_opt_work_buffers[2]

    @property
    def _nvme_para_receive_buffer(self):
        return self._nvme_para_buffers[0]

    @property
    def _nvme_para_offload_buffer(self):
        return self._nvme_para_buffers[1]

    @property
    def _nvme_grad_receive_buffer(self):
        return self._nvme_grad_buffers[0]

    @property
    def _nvme_grad_offload_buffer(self):
        return self._nvme_grad_buffers[1]

    def _is_invalid_unit(self, unit_index: int):
        return unit_index < 0 or unit_index >= self._num_units

    def _is_invalid_micro_batch(self, micro_batch_index: int):
        return micro_batch_index < 0 or \
            micro_batch_index >= self._micro_batch_per_rank

    def _is_invalid_task(self, unit_index: int, micro_batch_index: int):
        return self._is_invalid_unit(unit_index) or \
            self._is_invalid_micro_batch(micro_batch_index)

    def initialize(
        self,
        cpu_optimizer: FlexTrainCPUOptimizer,
        opt_state_per_element: int
    ):
        # 0. Before initialization:
        # Ensure that the parameter coordinator is initialized.
        para = get_para_coordinator()
        assert para.is_initialized, (
            "Parameter coordinator must be initialized before init_optimizer."
        )
        self._para = para

        # Link the optimizer to the coordinator.
        self._optimizer = cpu_optimizer
        self._inflight_optimizer_step = DummyHandle()

        # Link to the same NVMe swapper as the parameter coordinator.
        self._nvme_swapper = para._nvme_swapper
        # Use the same data stream as the parameter & inter-layer coordinator.
        self._data_stream = get_data_stream()

        # 1. Set the configuration for the optimizer.
        self._num_units = para.num_units
        self._unit_parameters = para._unit_parameters
        # Master parameters are also regarded as a optimizer state.
        self._opt_state_per_element = opt_state_per_element + 1

        config = get_flextrain_config()

        assert config.split_ratio.optimizer[0] == 0., (
            "FlexTrain optimizer currently does not support GPU optimizer. "
            "Please set the GPU optimizer ratio to 0. "
        )
        # Drop the GPU optimizer ratio.
        opts_cpu_nvme_ratio = config.split_ratio.optimizer[1:]

        # Configuration for mixed precision.
        device_dtype = config.mixed_precision.device_dtype
        gradacc_dtype = config.mixed_precision.gradacc_dtype
        master_dtype = config.mixed_precision.master_dtype
        self._gradacc_dtype_incompatible = device_dtype != gradacc_dtype

        # Configuration for optimizer partition.
        self._alpha = config.split_ratio.alpha[0]
        self._unit_numel = para._aligned_unit_numel
        self._micro_batch_per_rank = para._micro_batch_per_rank

        self._mb_gpu_para_alpha_splits = para._micro_batch_gpu_alpha_splits
        self._mb_cpu_para_alpha_splits = para._micro_batch_cpu_alpha_splits
        self._mb_nvme_para_alpha_splits = para._micro_batch_nvme_alpha_splits

        # Calculate the parameter numels of forward and backward.
        self._mb_fwd_numel = \
            self._mb_gpu_para_alpha_splits[0] + \
            self._mb_cpu_para_alpha_splits[0] + \
            self._mb_nvme_para_alpha_splits[0]
        self._unit_fwd_numel = self._mb_fwd_numel * self._micro_batch_per_rank

        self._mb_bwd_numel = \
            self._mb_gpu_para_alpha_splits[1] + \
            self._mb_cpu_para_alpha_splits[1] + \
            self._mb_nvme_para_alpha_splits[1]
        self._unit_bwd_numel = self._mb_bwd_numel * self._micro_batch_per_rank

        max_numel = max(self._unit_fwd_numel, self._unit_bwd_numel)

        # Calculate the numel of forward and backward optimizer states.
        self._mb_fwd_opt_numel = self._mb_fwd_numel * \
            self._opt_state_per_element
        self._unit_fwd_opt_numel = self._unit_fwd_numel * \
            self._opt_state_per_element

        self._mb_bwd_opt_numel = self._mb_bwd_numel * \
            self._opt_state_per_element
        self._unit_bwd_opt_numel = self._unit_bwd_numel * \
            self._opt_state_per_element

        # How to split the micro-batch parameters.
        self._mb_para_splits = [
            *self._mb_cpu_para_alpha_splits,
            *self._mb_gpu_para_alpha_splits,
            *self._mb_nvme_para_alpha_splits
        ]
        # How to split the forward part of the micro-batch parameters.
        self._mb_fwd_para_splits = [
            self._mb_cpu_para_alpha_splits[0],
            self._mb_gpu_para_alpha_splits[0],
            self._mb_nvme_para_alpha_splits[0]
        ]
        # How to split the backward part of the micro-batch parameters.
        self._mb_bwd_para_splits = [
            self._mb_cpu_para_alpha_splits[1],
            self._mb_gpu_para_alpha_splits[1],
            self._mb_nvme_para_alpha_splits[1]
        ]

        # How to split the forward optimizer states across devices.
        self._mb_fwd_opt_splits = get_split_numels(
            self._mb_fwd_opt_numel,
            opts_cpu_nvme_ratio, num_levels=2
        )
        self._unit_fwd_opt_splits = [
            numel * self._micro_batch_per_rank
            for numel in self._mb_fwd_opt_splits
        ]
        # How to split the backward optimizer states across devices.
        self._mb_bwd_opt_splits = get_split_numels(
            self._mb_bwd_opt_numel,
            opts_cpu_nvme_ratio, num_levels=2
        )
        self._unit_bwd_opt_splits = [
            numel * self._micro_batch_per_rank
            for numel in self._mb_bwd_opt_splits
        ]

        # NVMe group for optimizer.
        self._opt_nvme_group = NVMeGroup([
            self._unit_fwd_opt_splits[1],
            self._unit_bwd_opt_splits[1]
        ])
        # End of configuration.

        # 2. Allocate memory for the optimizer.
        # Used for accumulating / transferring backward gradients.
        self._gpu_bwd_grad_buffers = RotateContainer(
            allocate_memory_chunks(
                self._unit_numel, 2, gradacc_dtype,
                torch.cuda.current_device()
            )
        )

        # If gradacc_dtype is different from device_dtype,
        # we need an extra buffer for backward gradients.
        if self._gradacc_dtype_incompatible:
            self._gpu_bwd_extra_grads = torch.empty(
                self._unit_numel, dtype=device_dtype,
                device=torch.cuda.current_device()
            )

        # Used for receiving gradients / working with optimizer.
        self._cpu_opt_grad_buffers = RotateContainer(
            allocate_memory_chunks(
                max_numel, 2, master_dtype, torch.device('cpu')
            )
        )

        # Optimizer working buffer.
        self._cpu_opt_work_buffers = RotateContainer(
            allocate_memory_chunks(
                max_numel * self._opt_state_per_element,
                3, master_dtype, torch.device('cpu')
            )
        )

        # NVMe parameter offload buffer for backward NVMe parameters.
        self._nvme_para_buffers = RotateContainer(
            allocate_memory_chunks(
                self._mb_bwd_para_splits[2] * self._micro_batch_per_rank,
                2, device_dtype, torch.device('cpu')
            )
        )
        self._para_nvme_group = para._nvme_group

        # CPU optimizer base.
        self._cpu_opt_fwd_base = allocate_memory_chunks(
            self._unit_fwd_opt_splits[0], self._num_units,
            master_dtype, torch.device('cpu')
        ).zero_()
        self._cpu_opt_bwd_base = allocate_memory_chunks(
            self._unit_bwd_opt_splits[0], self._num_units,
            master_dtype, torch.device('cpu')
        ).zero_()
        # End of memory allocation.

        # 3. Initialize master parameters from device parameters.
        gpu_base = para._gpu_para_base
        cpu_base = para._cpu_para_base
        nvme_mem = para._nvme_available_paras
        nvme_fwd_mem, nvme_bwd_mem = torch.split(
            nvme_mem, para._unit_nvme_alpha_splits
        )
        nvme_fwd_base = torch.chunk(nvme_fwd_mem, self._micro_batch_per_rank)
        nvme_bwd_base = torch.chunk(nvme_bwd_mem, self._micro_batch_per_rank)
        temp_cpu_buffer: Tensor = self._cpu_opt_available_states

        dist.barrier()
        units = tqdm(
            range(self._num_units), desc="FlexTrain Opts. Init."
        ) if dist.get_rank() == 0 else range(self._num_units)
        for unit in units:
            # 1. Load the NVMe parameters.
            self._para_nvme_group.group_reload(
                FlexTrainDataID(Dtype.PARA, unit),
                nvme_mem, async_op=False
            )

            # 2. Create the forward optimizer states.
            # 2.1 Clear the temporary buffer.
            temp_cpu_buffer.zero_()
            fwd_tar_mem = temp_cpu_buffer[:self._unit_fwd_numel]
            fwd_tars = torch.chunk(fwd_tar_mem, self._micro_batch_per_rank)

            # 2.2 Copy the forward parameters.
            for micro_batch in range(self._micro_batch_per_rank):
                # Locate the source memory.
                gpu_src = gpu_base[unit][micro_batch]
                gpu_fwd_src, _ = torch.split(
                    gpu_src, self._mb_gpu_para_alpha_splits
                )
                cpu_src = cpu_base[unit][micro_batch]
                cpu_fwd_src, _ = torch.split(
                    cpu_src, self._mb_cpu_para_alpha_splits
                )
                nvme_fwd_src = nvme_fwd_base[micro_batch]

                # Locate the target memory.
                fwd_tar = fwd_tars[micro_batch]
                cpu_fwd_tar, gpu_fwd_tar, nvme_fwd_tar = torch.split(
                    fwd_tar, self._mb_fwd_para_splits
                )

                # Copy parameters from three sources.
                gpu_fwd_tar.copy_(gpu_fwd_src)
                cpu_fwd_tar.copy_(cpu_fwd_src)
                nvme_fwd_tar.copy_(nvme_fwd_src)

            # 2.3 Store the forward optimizer states.
            # Locate the source memory.
            fwd_src = temp_cpu_buffer[:self._unit_fwd_opt_numel]
            cpu_fwd_src, nvme_fwd_src = torch.split(
                fwd_src, self._unit_fwd_opt_splits
            )

            # Locate the target memory.
            cpu_fwd_tar = self._cpu_opt_fwd_base[unit]

            # Copy the optimizer states.
            cpu_fwd_tar.copy_(cpu_fwd_src)
            self._opt_nvme_group.single_offload(
                FlexTrainDataID(Dtype.OPTS, unit),
                nvme_fwd_src, index=0, async_op=False
            )
            # End of forward parameters.

            # 3. Create the backward optimizer states.
            # 3.1 Clear the temporary buffer.
            temp_cpu_buffer.zero_()
            bwd_tar_mem = temp_cpu_buffer[:self._unit_bwd_numel]
            bwd_tars = torch.chunk(bwd_tar_mem, self._micro_batch_per_rank)

            # 3.2 Copy the backward parameters.
            for micro_batch in range(self._micro_batch_per_rank):
                # Locate the source memory.
                gpu_src = gpu_base[unit][micro_batch]
                _, gpu_bwd_src = torch.split(
                    gpu_src, self._mb_gpu_para_alpha_splits
                )
                cpu_src = cpu_base[unit][micro_batch]
                _, cpu_bwd_src = torch.split(
                    cpu_src, self._mb_cpu_para_alpha_splits
                )
                nvme_bwd_src = nvme_bwd_base[micro_batch]

                # Locate the target memory.
                bwd_tar = bwd_tars[micro_batch]
                cpu_bwd_tar, gpu_bwd_tar, nvme_bwd_tar = torch.split(
                    bwd_tar, self._mb_bwd_para_splits
                )

                # Copy parameters from three sources.
                gpu_bwd_tar.copy_(gpu_bwd_src)
                cpu_bwd_tar.copy_(cpu_bwd_src)
                nvme_bwd_tar.copy_(nvme_bwd_src)

            # 3.3 Store the backward optimizer states.
            # Locate the source memory.
            bwd_src = temp_cpu_buffer[:self._unit_bwd_opt_numel]
            cpu_bwd_src, nvme_bwd_src = torch.split(
                bwd_src, self._unit_bwd_opt_splits
            )

            # Locate the target memory.
            cpu_bwd_tar = self._cpu_opt_bwd_base[unit]

            # Copy the optimizer states.
            cpu_bwd_tar.copy_(cpu_bwd_src)
            self._opt_nvme_group.single_offload(
                FlexTrainDataID(Dtype.OPTS, unit),
                nvme_bwd_src, index=1, async_op=False
            )
            # End of backward parameters.
        # End of optimizer state initialization.

    def _initialize_alpha_split(self):
        # If the coordinator is already initialized, return.
        if self._initialized:
            return
        self._initialized = True

        # Ensure that the parameter coordinator is initialized.
        para = get_para_coordinator()
        assert para.is_initialized, (
            "Parameter coordinator must be initialized."
        )
        # Ensure that the interlayer coordinator is initialized.
        interlayer = get_interlayer_coordinator()
        assert interlayer.is_initialized, (
            "Interlayer coordinator must be initialized."
        )

        # Figure out the numel each buffer can hold.
        cpu_grad_buffer_numel = interlayer._ckpt_numels[1]
        gpu_grad_buffer_numel = interlayer._ckpt_numels[0]

        # Total forward gradient numel.
        forward_grad_numel = self._mb_fwd_numel

        # Consider dtype incompatibility.
        config = get_flextrain_config()
        device_itemsize = config.mixed_precision.device_dtype.itemsize
        gradacc_itemsize = config.mixed_precision.gradacc_dtype.itemsize
        ratio = gradacc_itemsize // device_itemsize

        # Assign memory buffers and update plans.
        # Followings are the most complicated part of this repository.

        # cvtd = converted
        cvtd_cpu_grad_buffer_numel = cpu_grad_buffer_numel // ratio
        cvtd_gpu_grad_buffer_numel = gpu_grad_buffer_numel // ratio
        numels = [cvtd_cpu_grad_buffer_numel, cvtd_gpu_grad_buffer_numel]

        if forward_grad_numel < sum(numels[:1]):
            cpu_buffer_needed_numel = forward_grad_numel - sum(numels[:0])
            gpu_buffer_needed_numel = 0
            nvme_numel = 0
        elif forward_grad_numel < sum(numels[:2]):
            cpu_buffer_needed_numel = cvtd_cpu_grad_buffer_numel
            gpu_buffer_needed_numel = forward_grad_numel - sum(numels[:1])
            nvme_numel = 0
        else:
            cpu_buffer_needed_numel = cvtd_cpu_grad_buffer_numel
            gpu_buffer_needed_numel = cvtd_gpu_grad_buffer_numel
            nvme_numel = forward_grad_numel - sum(numels[:2])

        def _create_view(tensor: Tensor, numel: int, dtype: torch.dtype):
            return _convert_dtype_view(tensor, dtype)[..., :numel]

        # Create the gradient buffers.
        self._cpu_grad_buffer = _create_view(
            interlayer.cpu_ckpt_base, cpu_buffer_needed_numel,
            config.mixed_precision.gradacc_dtype
        )
        self._gpu_grad_buffer = _create_view(
            interlayer.gpu_ckpt_base, gpu_buffer_needed_numel,
            config.mixed_precision.gradacc_dtype
        )

        # How to reconstruct the forward gradients.
        self._forward_grad_splits = [
            cpu_buffer_needed_numel, gpu_buffer_needed_numel, nvme_numel
        ]

        # NVMe gradient buffers.
        self._nvme_grad_buffers = RotateContainer(
            allocate_memory_chunks(
                nvme_numel * self._micro_batch_per_rank, 2,
                config.mixed_precision.gradacc_dtype, torch.device('cpu')
            )
        )
        # End of assignment. Incredible!!!

        # Log the configuration.
        rank0_logger.info(
            "\n\n> "
            f"FlexTrain optimizer coordinator initialized "
            f"with configurations:\n"
            f"  - Parameter alpha split ratio (forward, backward): "
            f"({self._alpha}, {1 - self._alpha})\n"
            f"  - Parameter alpha split numels (forward, backward): "
            f"({self._mb_fwd_numel}, {self._mb_bwd_numel})\n"
            f"  - Micro-batch forward parameter split numels "
            f"(CPU, GPU, NVMe): {self._mb_fwd_para_splits}\n"
            f"  - Unit forward optimizer split numels (CPU, NVMe): "
            f"{self._unit_fwd_opt_splits}\n"
            f"  - Micro-batch backward parameter split numels "
            f"(CPU, GPU, NVMe): {self._mb_bwd_para_splits}\n"
            f"  - Unit backward optimizer split numels (CPU, NVMe): "
            f"{self._unit_bwd_opt_splits}\n"
            f"  - Forward gradient numel: {forward_grad_numel}\n"
            f"  - Gradient dtype itemsize / device dtype itemsize: {ratio}\n"
            f"  - Checkpoint borrowable numels (CPU, GPU): "
            f"({cvtd_cpu_grad_buffer_numel}, {cvtd_gpu_grad_buffer_numel})\n"
            f"  - Gradient buffer numels (CPU, GPU, NVMe): "
            f"{self._forward_grad_splits}\n"
        )

        self._grad_partition = allocate_memory_chunks(
            self._mb_fwd_numel + self._mb_bwd_numel, (self._num_units, self._micro_batch_per_rank),
            config.mixed_precision.gradacc_dtype, torch.cuda.current_device()
        )

    def _calculate_global_grad_norm(self):
        # Calculate the global gradient norm.
        global_grad_norm = torch.tensor([0.], device=torch.cuda.current_device())
        for unit in reversed(range(self._num_units)):
            global_grad_norm += self._grad_partition[unit].norm() ** 2
            dist.print_rank0(f"Rank {dist.get_rank()} layer {unit} grad norm: {global_grad_norm.item()}")
        # torch.save(self._grad_partition, f"/shared_ssd_storage/yikang/FlexTrain/logs/grad2/grad_norm_{dist.get_rank()}.pt")
        dist.all_reduce(global_grad_norm, op=dist.ReduceOp.SUM)
        dist.print_rank0(global_grad_norm.item())
        return global_grad_norm.item()

    def _rotate_buffers(self):
        self._gpu_bwd_grad_buffers.rotate()
        self._cpu_opt_grad_buffers.rotate()
        self._cpu_opt_work_buffers.rotate()
        self._nvme_grad_buffers.rotate()
        self._nvme_para_buffers.rotate()

    def _prepare_unit_grads(self, unit_index: int):
        """
        Prepare the gradients for the unit.
        The gradients are prepared in _gpu_available_grads.

        Args:
            unit_index (int): Index of the unit.

        Returns:
            None
        """
        # Check if the unit is valid.
        if self._is_invalid_unit(unit_index):
            return

        # Get the unit parameters.
        unit_paras = self._unit_parameters[unit_index]

        # Get the gradient buffer.
        if self._gradacc_dtype_incompatible:
            grad_buffer = self._gpu_bwd_extra_grads
        else:
            grad_buffer = self._gpu_bwd_receive_grads

        # Zero the receive buffer.
        torch.zero_(self._gpu_bwd_receive_grads)

        # Link the gradients.
        unit_paras.link_grad_to(grad_buffer)

    def _submit_transfer_grad(self, unit_index: int, micro_batch_index: int):
        """ Launch the async IO operation to transfer gradients. """
        # Return if the task is invalid.
        if self._is_invalid_task(unit_index, micro_batch_index):
            return lambda: None

        # Get the default stream.
        default_stream = torch.cuda.current_stream()

        def transfer_grads():
            # Locate the target memory.
            src_full_grads = torch.chunk(
                self._gpu_bwd_transfer_grads, self._micro_batch_per_rank
            )[micro_batch_index]
            mem_partition = torch.chunk(
                src_full_grads, dist.get_world_size()
            )[dist.get_rank()]

            # Synchronize with the default stream for the first unit.
            # Because there is no torch.cuda.synchronize() before.
            if unit_index == 0:
                default_stream.synchronize()

            # All-reduce the gradients.
            dist.reduce_scatter(
                mem_partition, src_full_grads,
                dist.ReduceOp.AVG
            )

            # Split the mem_partition.
            fwd_cpu, bwd_cpu, fwd_gpu, bwd_gpu, fwd_nvme, bwd_nvme = \
                torch.split(mem_partition, self._mb_para_splits)

            # Store forward gradients into buffers.
            copy_segments(
                [fwd_cpu, fwd_gpu, fwd_nvme],
                [
                    self._cpu_grad_buffer[unit_index][micro_batch_index],
                    self._gpu_grad_buffer[unit_index][micro_batch_index],
                    torch.chunk(
                        self._nvme_grad_receive_buffer,
                        self._micro_batch_per_rank
                    )[micro_batch_index]
                ]
            )

            # Move backward gradients into working buffer.
            opt_grad_tar = torch.chunk(
                self._cpu_opt_receive_grads[:self._unit_bwd_numel],
                self._micro_batch_per_rank
            )[micro_batch_index]
            bwd_cpu_tar, bwd_gpu_tar, bwd_nvme_tar = torch.split(
                opt_grad_tar, self._mb_bwd_para_splits
            )
            bwd_cpu_tar.copy_(bwd_cpu, non_blocking=True)
            bwd_gpu_tar.copy_(bwd_gpu, non_blocking=True)
            bwd_nvme_tar.copy_(bwd_nvme, non_blocking=True)

            self._grad_partition[unit_index][micro_batch_index].copy_(mem_partition)

        # Return the task, will be submitted to the data stream.
        return transfer_grads

    def _submit_transfer_opts(self, unit_index: int, forward: bool):
        """ Launch the async IO operation to transfer CPU optimizer states. """
        # Return if the task is invalid.
        if self._is_invalid_unit(unit_index):
            return

        # 1. Locate the source memory.
        cpu_src = self._cpu_opt_fwd_base if forward else self._cpu_opt_bwd_base
        cpu_src = cpu_src[unit_index]

        # 2. Locate the target memory.
        receive_buffer = self._cpu_opt_receive_states
        clip_numel = self._unit_fwd_opt_numel \
            if forward else self._unit_bwd_opt_numel
        receive_buffer = receive_buffer[:clip_numel]
        split_numel = self._unit_fwd_opt_splits \
            if forward else self._unit_bwd_opt_splits
        cpu_tar = torch.split(receive_buffer, split_numel)[0]

        # 3. Copy the source memory to the target memory.
        cpu_tar.copy_(cpu_src, non_blocking=True)

    def _submit_optimizer_step(self, unit_index: int, forward: bool):
        """ Launch the async IO operation to update CPU optimizer states. """
        # Return if the task is invalid.
        if self._is_invalid_unit(unit_index):
            return lambda: None

        # 1. Locate the optimizer states.
        opts_mem = self._cpu_opt_available_states
        clip_numel = self._unit_fwd_opt_numel \
            if forward else self._unit_bwd_opt_numel
        opts_mem = opts_mem[:clip_numel]
        optimizer_states = torch.chunk(opts_mem, self._opt_state_per_element)

        # 2. Locate the gradient memory.
        grad_mem = self._cpu_opt_available_grads
        clip_numel = self._unit_fwd_numel if forward else self._unit_bwd_numel
        gradients = grad_mem[:clip_numel]

        # TEMP:
        if not hasattr(self, "_zeros"):
            self._zeros = torch.zeros_like(gradients)

        # 3. Submit the optimizer step.
        self._inflight_optimizer_step = self._optimizer.submit_step(
            unit_index, optimizer_states[0], self._zeros, *optimizer_states[1:]
        )

    def _submit_update_opts(self, unit_index: int, forward: bool):
        """ Launch the async IO operation to update CPU optimizer states. """
        # Return if the task is invalid.
        if self._is_invalid_unit(unit_index):
            return

        # 1. Locate the source memory.
        writeback_buffer = self._cpu_opt_transfer_states
        clip_numel = self._unit_fwd_opt_numel \
            if forward else self._unit_bwd_opt_numel
        writeback_buffer = writeback_buffer[:clip_numel]
        split_numel = self._unit_fwd_opt_splits \
            if forward else self._unit_bwd_opt_splits
        cpu_src = torch.split(writeback_buffer, split_numel)[0]

        # 2. Locate the target memory.
        cpu_tar = self._cpu_opt_fwd_base if forward else self._cpu_opt_bwd_base
        cpu_tar = cpu_tar[unit_index]

        # 3. Copy the source memory to the target memory.
        cpu_tar.copy_(cpu_src, non_blocking=True)

    def _submit_update_para(
        self,
        unit_index: int,
        micro_batch_index: int,
        forward: bool
    ):
        """ Launch the async IO operation to update parameters. """
        # Return if the task is invalid.
        if self._is_invalid_task(unit_index, micro_batch_index):
            return lambda: None

        # 1. Locate the source memory.
        updated_para = self._cpu_opt_transfer_states
        clip_numel = self._unit_fwd_numel if forward else self._unit_bwd_numel
        updated_para = updated_para[:clip_numel]
        updated_para = torch.chunk(
            updated_para, self._micro_batch_per_rank
        )[micro_batch_index]
        split_numel = self._mb_fwd_para_splits \
            if forward else self._mb_bwd_para_splits
        cpu_src, gpu_src, nvme_src = torch.split(updated_para, split_numel)

        # 2. Locate the target memory.
        cpu_para = self._para._cpu_para_base[unit_index][micro_batch_index]
        cpu_fwd_tar, cpu_bwd_tar = torch.split(
            cpu_para, self._mb_cpu_para_alpha_splits
        )
        cpu_tar = cpu_fwd_tar if forward else cpu_bwd_tar
        gpu_para = self._para._gpu_para_base[unit_index][micro_batch_index]
        gpu_fwd_tar, gpu_bwd_tar = torch.split(
            gpu_para, self._mb_gpu_para_alpha_splits
        )
        gpu_tar = gpu_fwd_tar if forward else gpu_bwd_tar
        nvme_para = self._para.nvme_forward_update_paras \
            if forward else self._nvme_para_receive_buffer
        nvme_tar = torch.chunk(
            nvme_para, self._micro_batch_per_rank
        )[micro_batch_index]

        # 3. Copy the source memory to the target memory.
        cpu_tar.copy_(cpu_src, non_blocking=True)
        nvme_tar.copy_(nvme_src, non_blocking=True)

        def _update_gpu_para():
            gpu_tar.copy_(gpu_src, non_blocking=True)

        # 4. Return the task, will be submitted to the data stream.
        return _update_gpu_para

    def pre_micro_batch_forward(self, curr_task: LLMTask):
        # For the first iteration, the optimizer is not ready.
        if not self.is_initialized:
            return

    def pre_micro_batch_backward(self, curr_task: LLMTask):
        """ Submit tasks for the given micro-batch in backward pass. """
        # Zero the extra buffer if necessary.
        if self._gradacc_dtype_incompatible:
            torch.zero_(self._gpu_bwd_extra_grads)

        # Unpack the task.
        unit_index, micro_batch_index = curr_task.unit, curr_task.micro_batch

        # Trick: execute CUDA stream operations after recomputation.
        transfer_grad = self._submit_transfer_grad(
            unit_index + 1, micro_batch_index
        )
        update_para = self._submit_update_para(
            unit_index + 3, micro_batch_index, forward=False
        )

        # Submit CUDA stream tasks to the post-recomputation function.
        def _pre_micro_batch_tasks():
            self._data_stream.submit(transfer_grad)
            self._data_stream.submit(update_para)
            self._data_stream.execute()
        set_post_recomputation_function(_pre_micro_batch_tasks)

    def post_micro_batch_backward(self, curr_task: LLMTask):
        """ Conduct post-processing after the backward of the micro-batch. """
        # If the gradients are not compatible with the device dtype,
        # we need explicitly accumulate the gradients.
        if self._gradacc_dtype_incompatible:
            gradacc_buffer = self._gpu_bwd_receive_grads
            gradacc_buffer += self._gpu_bwd_extra_grads

    def pre_unit_forward(self, unit_index: int):
        # If the optimizer is not initialized, return.
        if not self.is_initialized:
            return

        # Rotate the buffers.
        self._rotate_buffers()

    def pre_unit_backward(self, unit_index: int):
        """ Prepare the unit for backward pass.

        Functions:
        1. Ensure the availability of the gradients buffer.

        Args:
            unit_index (int): Index of the unit.

        Returns:
            None
        """
        # Check if the unit is valid.
        if self._is_invalid_unit(unit_index):
            return

        # Synchrnoize the inflight optimizer step.
        self._inflight_optimizer_step.wait()

        # Rotate the buffers.
        self._rotate_buffers()

        self._submit_transfer_opts(unit_index + 1, forward=False)
        self._submit_optimizer_step(unit_index + 2, forward=False)
        self._submit_update_opts(unit_index + 3, forward=False)

        # Prepare the gradient buffer.
        self._prepare_unit_grads(unit_index)

    def warmup_backward_pipeline(self):
        # Complete the initialization.
        self._initialize_alpha_split()
        # Inform the parameter coordinator that future updates are coming.
        self._para.parameter_updated = True

    def clear_backward_pipeline(self):
        """
        Cleanup the backward pipeline.
        """
        # Synchronize the inflight tasks.
        self._inflight_optimizer_step.wait()
        self._nvme_swapper.synchronize()
        self._data_stream.synchronize()

        # Rotate the buffers.
        self._rotate_buffers()

        # Conduct the optimizer step of the first unit.
        for mb in reversed(range(self._micro_batch_per_rank)):
            self._submit_transfer_grad(0, mb)()
        self._data_stream.execute()
        self._data_stream.synchronize()


_OPTS_COORDINATOR = FlexTrainOptsCoordinator()


def get_opts_coordinator():
    """
    Get the optimizer coordinator.

    Returns:
        FlexTrainOptsCoordinator: The optimizer coordinator.
    """
    return _OPTS_COORDINATOR
