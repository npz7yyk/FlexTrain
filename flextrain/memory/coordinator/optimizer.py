import torch

from abc import ABC, abstractmethod
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass
from torch import Tensor
from tqdm import tqdm
from typing import Tuple, Dict

from flextrain.checkpointing import set_pre_backward_function
from flextrain.config import get_flextrain_config
from flextrain.memory import (
    FlexTrainDataType as Dtype,
    FlexTrainDataID,
    RotateContainer,
    get_split_numels,
    allocate_memory_chunks,
    get_data_stream
)
from flextrain.memory.coordinator import (
    get_para_coordinator,
    get_interlayer_coordinator
)
from flextrain.memory.nvme_swapper import NVMeGroup
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
        self._future: Tuple[int, int, Future] = None
        self._unit_group_map = unit_group_map

        # Initialize the state for each unit.
        if not hasattr(self, "state"):
            self.state = defaultdict(dict)
        for unit in unit_group_map:
            self.state[unit] = {STEP_KEY: 0}

    @abstractmethod
    def _step(self, step: int, args: Dict, opt_target: OptTarget):
        pass

    def synchronize_micro_batch_step(
        self,
        unit_index: int,
        micro_batch_index: int
    ):
        if self._future is None:
            return

        last_unit_index, last_micro_batch_index, future = self._future
        assert unit_index == last_unit_index
        assert micro_batch_index == last_micro_batch_index
        future.result()
        self._future = None

    def update_state(self):
        for unit in self._unit_group_map:
            self.state[unit][STEP_KEY] += 1

    def _submit_micro_batch_step(
        self,
        unit_index: int,
        micro_batch_index: int,
        opt_target: OptTarget
    ):
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
        self._future = (unit_index, micro_batch_index, future)

    @abstractmethod
    def submit_micro_batch_step(
        self, unit_index: int, micro_batch_index: int,
        para: torch.Tensor, grad: torch.Tensor,
        *args, **kwargs
    ):
        pass


def _convert_dtype_view(
    tensor: Tensor,
    target_dtype: torch.dtype
) -> Tensor:
    if tensor.numel() == 0:
        assert tensor.shape[-1] == 0
        return torch.empty_like(tensor, dtype=target_dtype)
    return tensor.view(target_dtype)


class FlexTrainOptCoordinator:

    def __init__(self):
        # Lazy initialization of optimizer coordinator.
        self._initialized = False

    @property
    def is_initialized(self):
        return self._initialized

    @property
    def _gpu_bwd_receive_grads(self):
        return self._gpu_bwd_grads_buffer[0]

    @property
    def _gpu_bwd_transfer_grads(self):
        return self._gpu_bwd_grads_buffer[1]

    @property
    def _cpu_opt_receive_grads(self):
        return self._cpu_opt_grads_buffer[0]

    @property
    def _cpu_opt_available_grads(self):
        return self._cpu_opt_grads_buffer[1]

    @property
    def _cpu_opt_receive_states(self):
        return self._cpu_opt_work_buffer[0]

    @property
    def _cpu_opt_available_states(self):
        return self._cpu_opt_work_buffer[1]

    @property
    def _cpu_opt_transfer_states(self):
        return self._cpu_opt_work_buffer[2]

    @property
    def _cpu_grad_receive_buffer(self):
        return self._nvme_grad_buffers[0]

    @property
    def _cpu_grad_offload_buffer(self):
        return self._nvme_grad_buffers[1]

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
        # Ensure that the optimizer coordinator is not initialized yet.
        assert not self._initialized, (
            "Optimizer coordinator is already initialized."
        )
        self._initialized = True
        self._finalized = False

        # Link the optimizer to the coordinator.
        self._optimizer = cpu_optimizer

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
        self._unit_gpu_para_alpha_splits = para._unit_gpu_alpha_splits
        self._unit_cpu_para_alpha_splits = para._unit_cpu_alpha_splits
        self._unit_nvme_para_alpha_splits = para._unit_nvme_alpha_splits

        self._forward_numel = \
            self._mb_gpu_para_alpha_splits[0] + \
            self._mb_cpu_para_alpha_splits[0] + \
            self._mb_nvme_para_alpha_splits[0]
        self._backward_numel = \
            self._mb_gpu_para_alpha_splits[1] + \
            self._mb_cpu_para_alpha_splits[1] + \
            self._mb_nvme_para_alpha_splits[1]

        self._forward_para_splits = [
            self._mb_cpu_para_alpha_splits[0],
            self._mb_gpu_para_alpha_splits[0],
            self._mb_nvme_para_alpha_splits[0]
        ]
        self._backward_para_splits = [
            self._mb_cpu_para_alpha_splits[1],
            self._mb_gpu_para_alpha_splits[1],
            self._mb_nvme_para_alpha_splits[1]
        ]

        self._forward_opt_splits = get_split_numels(
            self._forward_numel * self._opt_state_per_element,
            opts_cpu_nvme_ratio, num_levels=2
        )
        self._backward_opt_splits = get_split_numels(
            self._backward_numel * self._opt_state_per_element,
            opts_cpu_nvme_ratio, num_levels=2
        )

        # NVMe group for optimizer.
        self.nvme_group = NVMeGroup([
            self._forward_opt_splits[1],
            self._backward_opt_splits[1]
        ])
        # End of configuration.

        # 2. Allocate memory for the optimizer.
        # Used for accumulating / transferring backward gradients.
        self._gpu_bwd_grads_buffer = RotateContainer(
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
        self._cpu_opt_grads_buffer = RotateContainer(
            allocate_memory_chunks(
                max(self._forward_numel, self._backward_numel),
                (2, self._opt_state_per_element),
                device_dtype, torch.device('cpu')
            )
        )

        # Optimizer working buffer.
        self._cpu_opt_work_buffer = RotateContainer(
            allocate_memory_chunks(
                max(self._forward_numel, self._backward_numel),
                (3, self._opt_state_per_element),
                master_dtype, torch.device('cpu')
            )
        )

        # CPU optimizer base.
        self._cpu_opt_fwd_base = allocate_memory_chunks(
            self._forward_opt_splits[0], (
                self._num_units,
                self._micro_batch_per_rank,
                self._opt_state_per_element
            ),
            master_dtype, torch.device('cpu')
        ).zero_()
        self._cpu_opt_bwd_base = allocate_memory_chunks(
            self._backward_opt_splits[0], (
                self._num_units,
                self._micro_batch_per_rank,
                self._opt_state_per_element
            ),
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
        para_nvme_group = para._nvme_group

        temp_cpu_buffer: Tensor = self._cpu_opt_available_states
        dist.barrier()
        units = tqdm(
            range(self._num_units),
            desc="FlexTrain Opt. Init."
        ) if dist.get_rank() == 0 else range(self._num_units)
        for unit in units:
            para_nvme_group.group_reload(
                FlexTrainDataID(unit, Dtype.PARA),
                nvme_mem, async_op=False
            )
            for micro_batch in range(self._micro_batch_per_rank):
                gpu_src = gpu_base[unit][micro_batch]
                gpu_fwd_src, gpu_bwd_src = torch.split(
                    gpu_src, self._mb_gpu_para_alpha_splits
                )
                cpu_src = cpu_base[unit][micro_batch]
                cpu_fwd_src, cpu_bwd_src = torch.split(
                    cpu_src, self._mb_cpu_para_alpha_splits
                )
                nvme_fwd_src = nvme_fwd_base[micro_batch]
                nvme_bwd_src = nvme_bwd_base[micro_batch]

                # 1. Copy the forward parameters.
                temp_cpu_buffer.zero_()

                # Locate the target memory.
                tar_fwd_mem = temp_cpu_buffer[0][:self._forward_numel]
                cpu_fwd_tar, gpu_fwd_tar, nvme_fwd_tar = torch.split(
                    tar_fwd_mem, self._forward_para_splits
                )

                # Copy parameters from three sources.
                gpu_fwd_tar.copy_(gpu_fwd_src)
                cpu_fwd_tar.copy_(cpu_fwd_src)
                nvme_fwd_tar.copy_(nvme_fwd_src)

                # Locate the source memory.
                fwd_src = temp_cpu_buffer.flatten()[
                    :self._forward_numel * self._opt_state_per_element
                ]
                cpu_fwd_src, nvme_fwd_src = torch.split(
                    fwd_src, self._forward_opt_splits
                )

                # Locate the target memory.
                cpu_fwd_tar = self._cpu_opt_fwd_base[unit][micro_batch]

                # Copy the parameters.
                cpu_fwd_tar.copy_(cpu_fwd_src)
                self.nvme_group.single_offload(
                    prefix=FlexTrainDataID(unit, Dtype.OPTS),
                    tensor=nvme_fwd_src,
                    index=0,
                    async_op=False
                )
                # End of forward parameters.

                # 2. Copy the backward parameters.
                temp_cpu_buffer.zero_()

                # Locate the target memory.
                tar_bwd_mem = temp_cpu_buffer[0][:self._backward_numel]
                cpu_bwd_tar, gpu_bwd_tar, nvme_bwd_tar = torch.split(
                    tar_bwd_mem, self._backward_para_splits
                )

                # Copy parameters from three sources.
                gpu_bwd_tar.copy_(gpu_bwd_src)
                cpu_bwd_tar.copy_(cpu_bwd_src)
                nvme_bwd_tar.copy_(nvme_bwd_src)

                # Locate the source memory.
                bwd_src = temp_cpu_buffer.flatten()[
                    :self._backward_numel * self._opt_state_per_element
                ]
                cpu_bwd_src, nvme_bwd_src = torch.split(
                    bwd_src, self._backward_opt_splits
                )

                # Locate the target memory.
                cpu_bwd_tar = self._cpu_opt_bwd_base[unit][micro_batch]

                # Copy the parameters.
                cpu_bwd_tar.copy_(cpu_bwd_src)
                self.nvme_group.single_offload(
                    prefix=FlexTrainDataID(unit, Dtype.OPTS),
                    tensor=nvme_bwd_src,
                    index=1,
                    async_op=False
                )
                # End of backward parameters.
        # End of initialization.

        # Last task handles.
        self._last_opt_transfer_task: Tuple[int, int] = None
        self._last_opt_step_task: Tuple[int, int] = None
        self._last_opt_writeback_task: Tuple[int, int] = None

    def finalize_alpha_split(self):
        # If the coordinator is already finalized, return.
        if self._finalized:
            return
        self._finalized = True

        # Ensure that the parameter coordinator is initialized.
        para = get_para_coordinator()
        assert para.is_initialized, (
            "Parameter coordinator must be initialized before init_optimizer."
        )
        # Ensure that the interlayer coordinator is initialized.
        interlayer = get_interlayer_coordinator()
        assert interlayer.is_initialized, (
            "Interlayer coordinator must be initialized before init_optimizer."
        )

        # Figure out the numel each buffer can hold.
        cpu_grad_buffer1_numel = interlayer._ckpt_numels[1]

        bwd_cpu_opt_cpu_para_overlap = min(
            self._backward_opt_splits[0],
            self._mb_cpu_para_alpha_splits[1]
        )
        # Fwd CPU buffer + bwd CPU overlap.
        cpu_grad_buffer2_numel = \
            self._mb_cpu_para_alpha_splits[0] + \
            bwd_cpu_opt_cpu_para_overlap

        gpu_grad_buffer1_numel = interlayer._ckpt_numels[0]

        bwd_cpu_opt_gpu_para_overlap = min(
            self._backward_opt_splits[0] - bwd_cpu_opt_cpu_para_overlap,
            self._mb_gpu_para_alpha_splits[1]
        )
        # Fwd GPU buffer + bwd GPU overlap.
        gpu_grad_buffer2_numel = \
            self._mb_gpu_para_alpha_splits[0] + \
            bwd_cpu_opt_gpu_para_overlap

        # Total forward gradient numel.
        forward_grad_numel = self._forward_numel

        # Consider dtype incompatibility.
        config = get_flextrain_config()
        device_itemsize = config.mixed_precision.device_dtype.itemsize
        gradacc_itemsize = config.mixed_precision.gradacc_dtype.itemsize
        ratio = gradacc_itemsize // device_itemsize

        # Assign memory buffers and update plans.
        # Followings are the most complicated part of this repository.

        # cvtd = converted
        cvtd_cpu_grad_buffer1_numel = cpu_grad_buffer1_numel // ratio
        cvtd_cpu_grad_buffer2_numel = cpu_grad_buffer2_numel // ratio
        cvtd_gpu_grad_buffer1_numel = gpu_grad_buffer1_numel // ratio
        cvtd_gpu_grad_buffer2_numel = gpu_grad_buffer2_numel // ratio
        numels = [
            cvtd_cpu_grad_buffer1_numel, cvtd_cpu_grad_buffer2_numel,
            cvtd_gpu_grad_buffer1_numel, cvtd_gpu_grad_buffer2_numel
        ]

        cpu_buffer1_needed_numel = cvtd_cpu_grad_buffer1_numel
        cpu_buffer2_needed_numel = cvtd_cpu_grad_buffer2_numel
        gpu_buffer1_needed_numel = cvtd_gpu_grad_buffer1_numel
        gpu_buffer2_needed_numel = cvtd_gpu_grad_buffer2_numel
        nvme_numel = forward_grad_numel - sum(numels[:4])
        if forward_grad_numel < sum(numels[:1]):
            cpu_buffer1_needed_numel = forward_grad_numel - sum(numels[:0])
            cpu_buffer2_needed_numel = 0
            gpu_buffer1_needed_numel = 0
            gpu_buffer2_needed_numel = 0
            nvme_numel = 0
        elif forward_grad_numel < sum(numels[:2]):
            cpu_buffer2_needed_numel = forward_grad_numel - sum(numels[:1])
            gpu_buffer1_needed_numel = 0
            gpu_buffer2_needed_numel = 0
            nvme_numel = 0
        elif forward_grad_numel < sum(numels[:3]):
            gpu_buffer1_needed_numel = forward_grad_numel - sum(numels[:2])
            gpu_buffer2_needed_numel = 0
            nvme_numel = 0
        elif forward_grad_numel < sum(numels[:4]):
            gpu_buffer2_needed_numel = forward_grad_numel - sum(numels[:3])
            nvme_numel = 0
        else:
            pass

        def _create_view(tensor: Tensor, numel: int, dtype: torch.dtype):
            if isinstance(tensor, list):
                assert all(isinstance(t, Tensor) for t in tensor)
                return [_create_view(t, numel, dtype) for t in tensor]
            return _convert_dtype_view(tensor[..., :numel], dtype)

        self._cpu_grad_buffer1 = _create_view(
            interlayer.cpu_ckpt_base, cpu_buffer1_needed_numel,
            config.mixed_precision.gradacc_dtype
        )
        self._cpu_grad_buffer2 = _create_view(
            para._cpu_para_base, cpu_buffer2_needed_numel,
            config.mixed_precision.gradacc_dtype
        )
        self._gpu_grad_buffer1 = _create_view(
            interlayer.gpu_ckpt_base, gpu_buffer1_needed_numel,
            config.mixed_precision.gradacc_dtype
        )
        self._gpu_grad_buffer2 = _create_view(
            para._gpu_para_base, gpu_buffer2_needed_numel,
            config.mixed_precision.gradacc_dtype
        )

        self._forward_mem_splits = [
            cpu_buffer1_needed_numel, cpu_buffer2_needed_numel,
            gpu_buffer1_needed_numel, gpu_buffer2_needed_numel,
            nvme_numel
        ]
        self._backward_update_splits = [
            # TODO: update the backward update splits.
        ]
        self._nvme_grad_buffers = RotateContainer(
            allocate_memory_chunks(
                nvme_numel, 2, config.mixed_precision.gradacc_dtype,
                torch.device('cpu')
            )
        )
        # End of assignment. Incredible!!!

        # Log the configuration.
        rank0_logger.info(
            "\n\n> "
            f"FlexTrain optimizer coordinator initialized "
            f"with configurations:\n"
            f"  - Parameter alpha split ratio (forward, backward): "
            f"({self._alpha}, {1. - self._alpha})\n"
            f"  - Parameter alpha split numels (forward, backward): "
            f"({self._forward_numel}, {self._backward_numel})\n"
            f"  - Forward parameter split numels (CPU, GPU, NVMe): "
            f"{self._forward_para_splits}\n"
            f"  - Forward optimizer split numels (CPU, NVMe): "
            f"{self._forward_opt_splits}\n"
            f"  - Backward parameter split numels (CPU, GPU, NVMe): "
            f"{self._backward_para_splits}\n"
            f"  - Backward optimizer split numels (CPU, NVMe): "
            f"{self._backward_opt_splits}\n"
            f"  - Forward gradient numel: {forward_grad_numel}\n"
            f"  - Forward gradient dtype ratio: {ratio}\n"
            f"  - Checkpoint borrowable numels (CPU1, GPU1): "
            f"({cvtd_cpu_grad_buffer1_numel}, {cvtd_gpu_grad_buffer1_numel})\n"
            f"  - Parameter borrowable numels (CPU2, GPU2): "
            f"({cvtd_cpu_grad_buffer2_numel}, {cvtd_gpu_grad_buffer2_numel})\n"
            f"  - Gradient buffer numels (CPU1, CPU2, GPU1, GPU2, NVMe): "
            f"{self._forward_mem_splits}\n"
        )

        exit()

    def _is_invalid_unit(self, unit_index: int):
        return unit_index < 0 or unit_index >= self._num_units

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

    def _submit_transfer_grads(self, unit_index: int, micro_batch_index: int):
        """ Launch the async IO operation to transfer gradients.

        Args:
            unit_index (int): Index of the unit.
            micro_batch_index (int): Index of the micro-batch.

        Returns:
            None
        """
        # Check if the unit is valid.
        if self._is_invalid_unit(unit_index):
            return

        # 1. Locate the target memory.
        # 2. Conduct all-reduce into tensor if necessary.
        default_stream = torch.cuda.current_stream()

        def transfer_grads():
            # Locate the target memory.
            src_full_grads = torch.chunk(
                self._gpu_bwd_transfer_grads, self._micro_batch_per_rank
            )[micro_batch_index]
            mem_partition = torch.chunk(
                src_full_grads, dist.get_world_size()
            )[dist.get_rank()]

            # Synchonize with the default stream for the first unit.
            # Because there is no torch.cuda.synchronize() before.
            if unit_index == 0:
                default_stream.synchronize()

            # All-reduce the gradients.
            dist.reduce_scatter(
                mem_partition, src_full_grads,
                dist.ReduceOp.AVG
            )

            self._cpu_opt_receive_grads.copy_(mem_partition, non_blocking=True)

        # Submit the task to the data stream.
        self._data_stream.submit(transfer_grads)

    def _submit_transfer_opts(
        self,
        unit_index: int,
        micro_batch_index: int,
        forward: bool = False
    ):
        """ Launch the async IO operation to transfer optimizer states.

        Args:
            unit_index (int): Index of the unit.
            micro_batch_index (int): Index of the micro-batch.

        Returns:
            None
        """
        # Check if the unit is valid.
        if self._is_invalid_unit(unit_index):
            return

        if forward:
            pass
        else:
            # dist.print_rank0(f"Submitting transfer opts unit {unit_index} micro_batch {micro_batch_index}")
            # TEMP: only use a single buffer for now.
            cpu_src = self._cpu_opt_base[unit_index][micro_batch_index]
            cpu_tar: Tensor = self._cpu_opt_receive_states
            cpu_tar.copy_(cpu_src, non_blocking=True)

        # Register the last gradient receive task.
        self._last_opt_transfer_task = (unit_index, micro_batch_index)

    def _synchonize_transfer_opts(self):
        # No task is needed for now.
        if self._last_opt_transfer_task is None:
            return

        rst = self._last_opt_transfer_task
        self._last_opt_transfer_task = None
        return rst

    def _submit_opt_step(self, unit_index: int, micro_batch_index: int):
        # dist.print_rank0(f"Submitting opt step unit {unit_index} micro_batch {micro_batch_index}")

        # 1. Locate the parameters / gradients / optimizer states.
        cpu_states = torch.chunk(
            self._cpu_opt_available_states, self._opt_state_per_element
        )
        cpu_paras = cpu_states[0]
        cpu_grads = self._cpu_opt_available_grads
        cpu_states = cpu_states[1:]

        # 2. Submit the optimizer step task.
        self._optimizer.submit_micro_batch_step(
            unit_index, micro_batch_index,
            cpu_paras, cpu_grads, *cpu_states
        )

        # 3. Update the last task.
        self._last_opt_step_task = (unit_index, micro_batch_index)

    def _synchonize_opt_step(self):
        # No task is needed for now.
        if self._last_opt_step_task is None:
            return

        # 1. Get the unit and micro-batch index.
        unit_index, micro_batch_index = self._last_opt_step_task

        # 2. Synchronize the optimizer states.
        self._optimizer.synchronize_micro_batch_step(
            unit_index, micro_batch_index
        )

        # 3. Update the last task.
        rst = self._last_opt_step_task
        self._last_opt_step_task = None
        return rst

    def _submit_writeback_opts(
        self,
        unit_index: int,
        micro_batch_index: int,
        forward: bool = False
    ):
        if forward:
            pass
        else:
            # dist.print_rank0(f"Submitting write back opt unit {unit_index} micro_batch {micro_batch_index}")
            # TEMP: only use a single buffer for now.
            cpu_src = self._cpu_opt_base[unit_index][micro_batch_index]
            cpu_tar: Tensor = self._cpu_opt_receive_states
            cpu_tar.copy_(cpu_src, non_blocking=True)

        # Update the last task.
        self._last_opt_writeback_task = (unit_index, micro_batch_index)

    def _synchonize_writeback_opts(self):
        # No task is needed for now.
        if self._last_opt_writeback_task is None:
            return

        self._last_opt_writeback_task = None

    def pre_micro_batch_backward(
        self, unit_index: int, micro_batch_index: int
    ):
        """ Submit tasks for the given micro-batch in backward pass.

        Args:
            unit_index (int): Index of the unit.
            micro_batch_index (int): Index of the micro-batch.

        Returns:
            None
        """
        # Check if the unit is valid.
        if self._is_invalid_unit(unit_index):
            return

        # Zero the extra buffer if necessary.
        if self._gradacc_dtype_incompatible:
            torch.zero_(self._gpu_bwd_extra_grads)

        # Synchonize the optimizer state transfer.
        last_transfer_opt = self._synchonize_transfer_opts()
        last_opt_step = self._synchonize_opt_step()
        self._synchonize_writeback_opts()

        # Rotate the buffers.
        self._cpu_opt_grads_buffer.rotate()
        self._cpu_opt_work_buffer.rotate()

        # Submit gradient transfer task.
        # It should be about the previous unit.
        # Trick: We submit and execute the task after recomputation.
        def pre_micro_batch_backward_task():
            self._submit_transfer_grads(unit_index + 1, micro_batch_index)
            self._data_stream.execute()
        set_pre_backward_function(pre_micro_batch_backward_task)

        # Submit the optimizer step task.
        if last_opt_step is not None:
            self._submit_writeback_opts(*last_opt_step)
        if last_transfer_opt is not None:
            self._submit_opt_step(*last_transfer_opt)
        self._submit_transfer_opts(unit_index + 1, micro_batch_index)

        dist.print_rank0()

    def post_micro_batch_backward(
        self, unit_index: int, micro_batch_index: int
    ):
        """ Conduct post-processing after the backward of the micro-batch.

        Args:
            unit_index (int): Index of the unit.
            micro_batch_index (int): Index of the micro-batch.

        Returns:
            None
        """
        # Check if the unit is valid.
        if self._is_invalid_unit(unit_index):
            return

        # If the gradients are not compatible with the device dtype,
        # we need explicitly accumulate the gradients.
        if self._gradacc_dtype_incompatible:
            gradacc_buffer = self._gpu_bwd_receive_grads
            gradacc_buffer += self._gpu_bwd_extra_grads

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

        self._gpu_bwd_grads_buffer.rotate()
        self._prepare_unit_grads(unit_index)

    def clear_backward_pipeline(self):
        """
        Cleanup the backward pipeline.
        """
        self._gpu_bwd_grads_buffer.rotate()
        for i in range(self._micro_batch_per_rank):
            self._submit_transfer_grads(0, i)
        self._data_stream.execute()
        self._data_stream.synchronize()


_OPT_COORDINATOR = FlexTrainOptCoordinator()


def get_opt_coordinator():
    """
    Get the optimizer coordinator.

    Returns:
        FlexTrainOptCoordinator: The optimizer coordinator.
    """
    return _OPT_COORDINATOR