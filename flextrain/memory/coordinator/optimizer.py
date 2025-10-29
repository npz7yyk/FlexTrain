import torch

from collections.abc import Iterable
from itertools import chain
from torch import Tensor
from tqdm import tqdm
from typing import Dict

from flextrain.checkpointing import set_post_recomputation_function
from flextrain.config import get_flextrain_config
from flextrain.memory import (
    FlexTrainDataType as Dtype,
    FlexTrainDataID,
    RotateContainer,
    get_split_numels,
    get_page_aligned_padding_numel,
    allocate_memory,
    allocate_memory_chunks,
    copy_segments,
    get_data_stream
)
from flextrain.memory.coordinator import (
    get_para_coordinator,
    get_interlayer_coordinator
)
from flextrain.memory.nvme_swapper import NVMeGroup
from flextrain.optimizer import FlexTrainOptimizer
from flextrain.param_group import (
    reshape_list,
    flatten_list,
    slice_segments,
    StepContextContainer
)
from flextrain.scheduler import LLMTask
from flextrain.utils import dist
from flextrain.utils.logging import rank0_logger


def _convert_dtype_view(
    tensor: Tensor,
    target_dtype: torch.dtype
) -> Tensor:
    if tensor.numel() == 0:
        assert tensor.shape[-1] == 0
        return torch.empty_like(tensor, dtype=target_dtype)
    return tensor.view(target_dtype)


class FlexTrainOptsCoordinator:

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
    def _cpu_opt_receive_grads(self) -> Tensor:
        return self._cpu_grad_buffers[0]

    @property
    def _cpu_opt_step_grads(self) -> Tensor:
        return self._cpu_grad_buffers[1]

    @property
    def _nvme_para_receive_buffer(self):
        padding_numel = self._para._unit_nvme_padding_numel
        if padding_numel:
            return self._nvme_para_buffers[0][:-padding_numel]
        else:
            return self._nvme_para_buffers[0]

    @property
    def _nvme_para_offload_buffer(self):
        return self._nvme_para_buffers[1]

    @property
    def _nvme_opts_prefetch_buffer(self) -> Tensor:
        return self._nvme_opts_buffers[0]

    @property
    def _nvme_opts_step_buffer(self) -> Tensor:
        return self._nvme_opts_buffers[1]

    @property
    def _nvme_opts_offload_buffer(self) -> Tensor:
        return self._nvme_opts_buffers[2]

    def _is_invalid_unit(self, unit_index: int):
        return unit_index < 0 or unit_index >= self._num_units

    def _is_invalid_micro_batch(self, micro_batch_index: int):
        return micro_batch_index < 0 or \
            micro_batch_index >= self._micro_batch_per_rank

    def _is_invalid_task(self, unit_index: int, micro_batch_index: int):
        return self._is_invalid_unit(unit_index) or \
            self._is_invalid_micro_batch(micro_batch_index)

    def initialize(self, optimizer: FlexTrainOptimizer):
        # 0. Before initialization:
        # Ensure that the parameter coordinator is initialized.
        para = get_para_coordinator()
        assert para.is_initialized, \
            "Parameter coordinator must be initialized before init_optimizer."
        self._para = para

        # Link the optimizer to the coordinator.
        self._optimizer = optimizer

        # Link to the same NVMe swapper as the parameter coordinator.
        self._nvme_swapper = para._nvme_swapper
        # Use the same data stream as the parameter & inter-layer coordinator.
        self._data_stream = get_data_stream()

        # 1. Set the configuration for the optimizer.
        self._auto_config = get_flextrain_config().auto_config
        self._num_units = para.num_units
        self._unit_parameters = para._unit_parameters
        config = get_flextrain_config()

        # Drop the GPU optimizer ratio.
        assert config.split_ratio.optimizer[0] == 0., (
            "FlexTrain optimizer currently does not support GPU optimizer. "
            "Please set the GPU optimizer ratio to 0. "
        )
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

        mb_gpu_alpha_splits = para._micro_batch_gpu_alpha_splits
        mb_cpu_alpha_splits = para._micro_batch_cpu_alpha_splits
        mb_nvme_alpha_splits = para._micro_batch_nvme_alpha_splits
        self._mb_gpu_alpha_splits = mb_gpu_alpha_splits
        self._mb_cpu_alpha_splits = mb_cpu_alpha_splits
        self._unit_nvme_alpha_splits = para._unit_nvme_alpha_splits

        self._mb_para_splits = [
            *mb_gpu_alpha_splits, *mb_cpu_alpha_splits, *mb_nvme_alpha_splits
        ]
        self._mb_fwd_para_splits = self._mb_para_splits[0::2]
        self._mb_bwd_para_splits = self._mb_para_splits[1::2]
        self._mb_fwd_numel = sum(self._mb_fwd_para_splits)
        self._mb_bwd_numel = sum(self._mb_bwd_para_splits)
        self._unit_fwd_numel = self._mb_fwd_numel * self._micro_batch_per_rank
        self._unit_bwd_numel = self._mb_bwd_numel * self._micro_batch_per_rank

        # How to split the forward master parameters across CPU/NVMe.
        self._unit_fwd_mpara_splits = get_split_numels(
            self._unit_fwd_numel, opts_cpu_nvme_ratio, master_dtype.itemsize, 2
        )
        # How to split the backward master parameters across CPU/NVMe.
        self._unit_bwd_mpara_splits = get_split_numels(
            self._unit_bwd_numel, opts_cpu_nvme_ratio, master_dtype.itemsize, 2
        )

        # 2. Allocate memory for the optimizer.
        # Used for accumulating / transferring backward gradients.
        self._gpu_bwd_grad_buffers = RotateContainer(allocate_memory_chunks(
            self._unit_numel, 2, gradacc_dtype, torch.cuda.current_device()
        ))

        # If gradacc_dtype is different from device_dtype,
        # we need an extra buffer for backward gradients.
        self._gpu_bwd_extra_grads = allocate_memory(
            self._unit_numel, device_dtype, torch.cuda.current_device()
        ) if self._gradacc_dtype_incompatible else None

        # Used for receiving gradients / working with optimizer.
        self._cpu_grad_buffers = RotateContainer(allocate_memory_chunks(
            self._unit_bwd_numel, 2, master_dtype, torch.device('cpu'),
        ))

        # Assumption:
        # All optimizer states have the same dtype as the master parameters.
        # May support complex optimizer states in the future.
        optimizer_state_per_element = \
            self._optimizer.optimizer_state_per_element

        # Allocate memory for master parameters + optimizer states.
        # +1 for master parameters.
        self._cpu_fwd_opts = allocate_memory_chunks(
            self._unit_fwd_mpara_splits[0],
            (self._num_units, 1 + optimizer_state_per_element),
            master_dtype, torch.device('cpu')
        )
        self._cpu_bwd_opts = allocate_memory_chunks(
            self._unit_bwd_mpara_splits[0],
            (self._num_units, 1 + optimizer_state_per_element),
            master_dtype, torch.device('cpu')
        )

        # Offloading buffer for backward NVMe parameters.
        self._nvme_para_buffers = RotateContainer(allocate_memory_chunks(
            para._nvme_group._numels[1],    # NVMe backward parameter numel
            2, device_dtype, torch.device('cpu')
        ))
        self._nvme_para_group = para._nvme_group

        # Offloading buffer for NVMe master parameters + optimizer states.
        self._nvme_fwd_opts_numel = \
            self._unit_fwd_mpara_splits[1] * (1 + optimizer_state_per_element)
        self._nvme_bwd_opts_numel = \
            self._unit_bwd_mpara_splits[1] * (1 + optimizer_state_per_element)
        self._nvme_opts_padding_numel = get_page_aligned_padding_numel(
            self._nvme_bwd_opts_numel, master_dtype.itemsize
        )
        self._nvme_bwd_offload_numel = \
            self._nvme_bwd_opts_numel + self._nvme_opts_padding_numel
        # Use three buffers for: prefetching, step, offloading.
        self._nvme_opts_buffers = RotateContainer(allocate_memory_chunks(
            self._nvme_bwd_offload_numel, 3, master_dtype, torch.device('cpu')
        ))
        self._nvme_opts_group = NVMeGroup([
            self._nvme_fwd_opts_numel, self._nvme_bwd_offload_numel
        ])

        # End of memory allocation.

        # 3. Initialize master parameters from device parameters.
        gpu_base = para._gpu_para_base
        cpu_base = para._cpu_para_base
        nvme_mem = para._nvme_inflight_paras
        nvme_fwd_mem, nvme_bwd_mem = \
            torch.split(nvme_mem, self._unit_nvme_alpha_splits)
        nvme_fwd_base = torch.chunk(nvme_fwd_mem, self._micro_batch_per_rank)
        nvme_bwd_base = torch.chunk(nvme_bwd_mem, self._micro_batch_per_rank)
        temp_para_buffer = self._cpu_opt_receive_grads
        temp_nvme_buffer = self._nvme_opts_offload_buffer

        dist.barrier()
        units = tqdm(
            range(self._num_units), desc="FlexTrain Optimizer Initialization"
        ) if dist.get_rank() == 0 else range(self._num_units)
        for unit in units:
            # 1. Load the NVMe parameters.
            self._nvme_para_group.group_reload(
                FlexTrainDataID(Dtype.PARA, unit),
                para._nvme_asyncio_paras, async_op=False
            )

            # 2. Create the forward optimizer states.
            # 2.1 Locate the target memory.
            temp_para_buffer.zero_()
            temp_nvme_buffer.zero_()
            fwd_tar_mem = temp_para_buffer[:self._unit_fwd_numel]
            fwd_tars = torch.chunk(fwd_tar_mem, self._micro_batch_per_rank)

            # 2.2 Locate the source memory and copy the parameters.
            for micro_batch in range(self._micro_batch_per_rank):
                # Locate the source memory.
                gpu_src = gpu_base[unit][micro_batch]
                gpu_fwd_src, _ = torch.split(gpu_src, mb_gpu_alpha_splits)
                cpu_src = cpu_base[unit][micro_batch]
                cpu_fwd_src, _ = torch.split(cpu_src, mb_cpu_alpha_splits)
                nvme_fwd_src = nvme_fwd_base[micro_batch]
                # Locate the target memory.
                fwd_tar = fwd_tars[micro_batch]
                gpu_fwd_tar, cpu_fwd_tar, nvme_fwd_tar = \
                    torch.split(fwd_tar, self._mb_fwd_para_splits)
                # Copy parameters from three sources.
                gpu_fwd_tar.copy_(gpu_fwd_src)
                cpu_fwd_tar.copy_(cpu_fwd_src)
                nvme_fwd_tar.copy_(nvme_fwd_src)

            # 2.3 Store the forward master parameters.
            # Locate the source memory.
            cpu_fwd_src, nvme_fwd_src = \
                torch.split(fwd_tar_mem, self._unit_fwd_mpara_splits)
            # Locate the target memory.
            cpu_fwd_tar = self._cpu_fwd_opts[unit][0]
            nvme_fwd_offload = temp_nvme_buffer[:self._nvme_fwd_opts_numel]
            nvme_fwd_tar = \
                nvme_fwd_offload.chunk(1 + optimizer_state_per_element)[0]
            # Copy the master parameters.
            cpu_fwd_tar.copy_(cpu_fwd_src)
            nvme_fwd_tar.copy_(nvme_fwd_src)
            # Offload the forward optimizer states.
            # Skip the NVMe offload if running in the auto-config mode.
            if not self._auto_config:
                self._nvme_opts_group.single_offload(
                    FlexTrainDataID(Dtype.OPTS, unit),
                    nvme_fwd_offload, index=0, async_op=False
                )
            # End of forward parameters initialization.

            # 3. Create the backward optimizer states.
            # 3.1 Locate the target memory.
            temp_para_buffer.zero_()
            temp_nvme_buffer.zero_()
            bwd_tar_mem = temp_para_buffer[:self._unit_bwd_numel]
            bwd_tars = torch.chunk(bwd_tar_mem, self._micro_batch_per_rank)

            # 3.2 Locate the source memory and copy the parameters.
            for micro_batch in range(self._micro_batch_per_rank):
                # Locate the source memory.
                gpu_src = gpu_base[unit][micro_batch]
                _, gpu_bwd_src = torch.split(gpu_src, mb_gpu_alpha_splits)
                cpu_src = cpu_base[unit][micro_batch]
                _, cpu_bwd_src = torch.split(cpu_src, mb_cpu_alpha_splits)
                nvme_bwd_src = nvme_bwd_base[micro_batch]
                # Locate the target memory.
                bwd_tar = bwd_tars[micro_batch]
                gpu_bwd_tar, cpu_bwd_tar, nvme_bwd_tar = \
                    torch.split(bwd_tar, self._mb_bwd_para_splits)
                # Copy parameters from three sources.
                gpu_bwd_tar.copy_(gpu_bwd_src)
                cpu_bwd_tar.copy_(cpu_bwd_src)
                nvme_bwd_tar.copy_(nvme_bwd_src)

            # 3.3 Store the backward optimizer states.
            # Locate the source memory.
            cpu_bwd_src, nvme_bwd_src = \
                torch.split(bwd_tar_mem, self._unit_bwd_mpara_splits)
            # Locate the target memory.
            cpu_bwd_tar = self._cpu_bwd_opts[unit][0]
            nvme_bwd_offload = temp_nvme_buffer[:self._nvme_bwd_offload_numel]
            buffer = temp_nvme_buffer[:self._nvme_bwd_opts_numel]
            nvme_bwd_tar = buffer.chunk(1 + optimizer_state_per_element)[0]
            # Copy the master parameters.
            cpu_bwd_tar.copy_(cpu_bwd_src)
            nvme_bwd_tar.copy_(nvme_bwd_src)
            # Offload the backward optimizer states.
            # Skip the NVMe offload if running in the auto-config mode.
            if not self._auto_config:
                self._nvme_opts_group.single_offload(
                    FlexTrainDataID(Dtype.OPTS, unit),
                    nvme_bwd_offload, index=1, async_op=False
                )
            # End of backward parameters initialization.
        # End of optimizer state initialization.

    def _initialize_alpha_split(self):
        # If the coordinator is already initialized, return.
        if self._initialized:
            return
        self._initialized = True

        # Ensure that the parameter coordinator is initialized.
        para = get_para_coordinator()
        assert para.is_initialized, \
            "Parameter coordinator must be initialized."
        # Ensure that the interlayer coordinator is initialized.
        interlayer = get_interlayer_coordinator()
        assert interlayer.is_initialized, \
            "Interlayer coordinator must be initialized."

        # Figure out the borrowable numel from each buffer.
        # The whole checkpoint buffer is borrowable.
        ckpt_buffer_numel = interlayer._ckpt_numels[1]
        # Only the forward part of the parameter buffer is borrowable.
        para_buffer_numel = para._micro_batch_cpu_alpha_splits[0]

        # Total forward gradient numel.
        forward_grad_numel = self._mb_fwd_numel

        # Consider dtype incompatibility.
        config = get_flextrain_config()
        device_itemsize = config.mixed_precision.device_dtype.itemsize
        gradacc_itemsize = config.mixed_precision.gradacc_dtype.itemsize
        ratio = gradacc_itemsize // device_itemsize

        # Assign memory buffers and update plans.
        ckpt_buffer_numel = ckpt_buffer_numel // ratio
        para_buffer_numel = para_buffer_numel // ratio
        if forward_grad_numel <= ckpt_buffer_numel:
            ckpt_buffer_needed_numel = forward_grad_numel
            para_buffer_needed_numel = 0
        elif forward_grad_numel <= ckpt_buffer_numel + para_buffer_numel:
            ckpt_buffer_needed_numel = ckpt_buffer_numel
            para_buffer_needed_numel = forward_grad_numel - ckpt_buffer_numel
        else:
            assert ratio >= 2
            mb_ckpt = interlayer._ckpt_numels[1]
            mb_para = self._mb_fwd_numel + self._mb_bwd_numel
            mb_cpu_para = sum(self._mb_cpu_alpha_splits)
            max_alpha = mb_ckpt / (ratio * mb_para - mb_cpu_para)
            raise NotImplementedError(
                "The forward gradient numel is too large to fit into the "
                "gradient buffers, consider:\n"
                "  - Increase the CPU checkpoint buffer numel, or\n"
                "  - Increase the CPU parameter buffer numel, or\n"
                "  - Reduce the alpha split ratio.\n"
                "Current alpha split ratio: {:.5f}\n".format(self._alpha) +
                "The maximum allowed alpha: {:.5f}".format(max_alpha)
            )

        def _create_view(tensor: Tensor, numel: int, dtype: torch.dtype):
            if not isinstance(tensor, Tensor):
                assert isinstance(tensor, Iterable)
                return [_create_view(t, numel, dtype) for t in tensor]
            return _convert_dtype_view(tensor, dtype)[..., :numel]

        # Create the gradient buffers.
        self._borrowed_ckpt_buffer = _create_view(
            interlayer.cpu_ckpt_base, ckpt_buffer_needed_numel,
            config.mixed_precision.gradacc_dtype
        )
        self._borrowed_para_buffer = _create_view(
            para._cpu_para_base, para_buffer_needed_numel,
            config.mixed_precision.gradacc_dtype
        )

        # How to reconstruct the forward gradients.
        fwd_grad_splits = [ckpt_buffer_needed_numel, para_buffer_needed_numel]

        # Create an extra gradient buffer for the last unit.
        # Otherwise, the borrowed parameter buffer is also the write target.
        # Avoid this by applying an offset of one unit.
        num_micro_batches = self._micro_batch_per_rank * dist.get_world_size()
        self._extra_grad_buffer = allocate_memory_chunks(
            para_buffer_needed_numel, num_micro_batches,
            config.mixed_precision.gradacc_dtype, torch.device('cpu')
        )

        # Create optimizer step contexts for each unit.
        self._unit_fwd_step_ctxts: Dict[int, StepContextContainer] = {}
        self._unit_bwd_step_ctxts: Dict[int, StepContextContainer] = {}
        unit_slices = self._mb_para_splits * num_micro_batches
        for unit in range(self._num_units):
            # Slice the segments for each unit.
            segment_groups = slice_segments(
                self._optimizer.unit_group_segments[unit], unit_slices
            )
            # micro_batch, rank, (GPU, CPU, NVMe), (fwd, bwd)
            segment_groups = reshape_list(
                segment_groups,
                (self._micro_batch_per_rank, dist.get_world_size(), 3, 2)
            )
            # layout: mb * (GPU, CPU, NVMe)
            fwd_segments = segment_groups[:, dist.get_rank(), :, 0]
            fwd_segments = flatten_list(fwd_segments.tolist())
            self._unit_fwd_step_ctxts[unit] = StepContextContainer(
                self._micro_batch_per_rank, fwd_segments,
                self._mb_fwd_para_splits * self._micro_batch_per_rank,
                self._unit_fwd_mpara_splits,
                fwd_grad_splits * self._micro_batch_per_rank
            )
            # layout: mb * (GPU, CPU, NVMe)
            bwd_segments = segment_groups[:, dist.get_rank(), :, 1]
            bwd_segments = flatten_list(bwd_segments.tolist())
            self._unit_bwd_step_ctxts[unit] = StepContextContainer(
                self._micro_batch_per_rank, bwd_segments,
                self._mb_bwd_para_splits * self._micro_batch_per_rank,
                self._unit_bwd_mpara_splits,
                [self._unit_bwd_numel]
            )
        # End of slice configuration and step context creation.

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
            f"(GPU, CPU, NVMe): {self._mb_fwd_para_splits}\n"
            f"  - Unit forward master parameter split numels (CPU, NVMe): "
            f"{self._unit_fwd_mpara_splits}\n"
            f"  - Micro-batch backward parameter split numels "
            f"(GPU, CPU, NVMe): {self._mb_bwd_para_splits}\n"
            f"  - Unit backward master parameter split numels (CPU, NVMe): "
            f"{self._unit_bwd_mpara_splits}\n"
            f"  - Forward gradient numel: {forward_grad_numel}\n"
            f"  - Gradient dtype itemsize / device dtype itemsize: {ratio}\n"
            f"  - Borrowable numels (checkpoint, parameter): "
            f"({ckpt_buffer_numel}, {para_buffer_numel})\n"
            f"  - Gradient borrowed numels (checkpoint, parameter): "
            f"({ckpt_buffer_needed_numel}, {para_buffer_needed_numel})\n"
        )

    def _rotate_buffers(self):
        self._gpu_bwd_grad_buffers.rotate()
        self._cpu_grad_buffers.rotate()
        self._nvme_opts_buffers.rotate()
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
        # IMPORTANT: This is delayed after torch.cuda.synchronize().
        self._data_stream.submit(
            lambda: torch.zero_(self._gpu_bwd_receive_grads),
            stream_execution=False
        )

        # Link the gradients.
        unit_paras.link_grad_to(grad_buffer)

    def _submit_transfer_grad(self, unit_index: int, micro_batch_index: int):
        """ Launch the async IO operation to transfer gradients. """
        # Return if the task is invalid.
        if self._is_invalid_task(unit_index, micro_batch_index):
            return lambda: None

        def transfer_grads():
            # Locate the target memory.
            src_full_grads = torch.chunk(
                self._gpu_bwd_transfer_grads, self._micro_batch_per_rank
            )[micro_batch_index]
            mem_partition = torch.chunk(
                src_full_grads, dist.get_world_size()
            )[dist.get_rank()]

            # All-reduce the gradients.
            dist.reduce_scatter(
                mem_partition, src_full_grads, dist.ReduceOp.AVG
            )

            # Split the mem_partition.
            fwd_gpu, bwd_gpu, fwd_cpu, bwd_cpu, fwd_nvme, bwd_nvme = \
                torch.split(mem_partition, self._mb_para_splits)

            # Store forward gradients into buffers.
            # The borrowed parameter buffer should have an offset of one unit.
            borrowed_ckpt_buffer = self._borrowed_ckpt_buffer[unit_index]
            borrowed_para_buffer = \
                self._extra_grad_buffer if unit_index == self._num_units - 1\
                else self._borrowed_para_buffer[unit_index + 1]
            copy_segments(
                [fwd_gpu, fwd_cpu, fwd_nvme],
                [
                    borrowed_ckpt_buffer[micro_batch_index],
                    borrowed_para_buffer[micro_batch_index]
                ]
            )

            # Move backward gradients into working buffer.
            opt_grad_tar = torch.chunk(
                self._cpu_opt_receive_grads, self._micro_batch_per_rank
            )[micro_batch_index]
            bwd_gpu_tar, bwd_cpu_tar, bwd_nvme_tar = \
                torch.split(opt_grad_tar, self._mb_bwd_para_splits)
            bwd_gpu_tar.copy_(bwd_gpu, non_blocking=True)
            bwd_cpu_tar.copy_(bwd_cpu, non_blocking=True)
            bwd_nvme_tar.copy_(bwd_nvme, non_blocking=True)

        # Return the task, will be submitted to the data stream.
        return transfer_grads

    def _submit_prefetch_opts(self, unit_index: int, forward: bool):
        """ Launch the async IO operation to prefetch CPU optimizer states. """
        # If running in the auto-config mode, return.
        if self._auto_config:
            return
        # Return if the task is invalid.
        if self._is_invalid_unit(unit_index):
            return

        # 1. Locate the target memory.
        numel = self._nvme_fwd_opts_numel \
            if forward else self._nvme_bwd_offload_numel
        nvme_tar = self._nvme_opts_prefetch_buffer[:numel]

        # 2. Launch the prefetch operation.
        self._nvme_opts_group.single_reload(
            FlexTrainDataID(Dtype.OPTS, unit_index),
            nvme_tar, index=0 if forward else 1, async_op=True
        )

    def _plan_optimizer_steps(
        self, unit_index: int, forward: bool, execute_now: bool = False
    ):
        """ Plan the optimizer steps for the given unit. """
        # Return if the task is invalid.
        if self._is_invalid_unit(unit_index):
            return

        # 1. Get the step context container.
        context_container = self._unit_fwd_step_ctxts[unit_index] \
            if forward else self._unit_bwd_step_ctxts[unit_index]

        # 2. Locate needed buffers.
        # Half parameters
        # GPU half parameters (no longer used)
        gpu_fwd_half_para, gpu_bwd_half_para = [], []
        for gpu_half_para in self._para._gpu_para_base[unit_index]:
            gpu_fwd_tar, gpu_bwd_tar = \
                torch.split(gpu_half_para, self._mb_gpu_alpha_splits)
            gpu_fwd_half_para.append(gpu_fwd_tar)
            gpu_bwd_half_para.append(gpu_bwd_tar)
        gpu_half_para = gpu_fwd_half_para if forward else gpu_bwd_half_para
        # CPU half parameters
        cpu_fwd_half_para, cpu_bwd_half_para = [], []
        for cpu_half_para in self._para._cpu_para_base[unit_index]:
            cpu_fwd_tar, cpu_bwd_tar = \
                torch.split(cpu_half_para, self._mb_cpu_alpha_splits)
            cpu_fwd_half_para.append(cpu_fwd_tar)
            cpu_bwd_half_para.append(cpu_bwd_tar)
        cpu_half_para = cpu_fwd_half_para if forward else cpu_bwd_half_para
        # NVMe half parameters
        nvme_half_para = self._para.nvme_forward_update_paras \
            if forward else self._nvme_para_receive_buffer
        nvme_half_para = \
            torch.chunk(nvme_half_para, self._micro_batch_per_rank)
        # Final half parameters, (3, micro_batch) -> (micro_batch * 3)
        half_parameter = list(chain.from_iterable(zip(
            gpu_half_para, cpu_half_para, nvme_half_para
        )))

        # Full parameters
        nvme_opts_numel = self._nvme_fwd_opts_numel \
            if forward else self._nvme_bwd_opts_numel
        nvme_opts = self._nvme_opts_step_buffer[:nvme_opts_numel]
        nvme_para, *nvme_opts = \
            nvme_opts.chunk(1 + self._optimizer.optimizer_state_per_element)
        full_parameter = [
            self._cpu_fwd_opts[unit_index][0] if forward else
            self._cpu_bwd_opts[unit_index][0], nvme_para
        ]

        # Gradients, (2, micro_batch) -> (micro_batch * 2)
        gradient = list(chain.from_iterable(zip(
            self._borrowed_ckpt_buffer[unit_index],
            self._borrowed_para_buffer[unit_index + 1]
            if unit_index < self._num_units - 1 else self._extra_grad_buffer
        ))) if forward else self._cpu_opt_step_grads

        # Optimizer states
        # Shape: (2, optimizer_state_per_element), need transpose
        optimizer_states = list(map(list, zip(
            self._cpu_fwd_opts[unit_index][1:] if forward else
            self._cpu_bwd_opts[unit_index][1:], nvme_opts
        )))

        # 3. Plan the steps.
        context_container.plan(
            half_parameter=half_parameter,
            full_parameter=full_parameter,
            gradient=gradient,
            optimizer_states=optimizer_states
        )

        # 4. Execute the plan now if required.
        if not execute_now:
            return
        for mb in range(self._micro_batch_per_rank):
            self._execute_optimizer_step(unit_index, mb, forward)

    def _execute_optimizer_step(
        self, unit_index: int, micro_batch_index: int, forward: bool
    ):
        """ Launch the async IO operation to update CPU optimizer states. """
        # Return if the task is invalid.
        if self._is_invalid_task(unit_index, micro_batch_index):
            return

        # Submit the optimizer step.
        context_container = self._unit_fwd_step_ctxts[unit_index] \
            if forward else self._unit_bwd_step_ctxts[unit_index]
        step_contexts = context_container[micro_batch_index]
        self._optimizer.cpu_optimizer.step(step_contexts)

    def _submit_offload_para(self, unit_index: int):
        """ Launch the async IO operation to offload parameters. """
        # Return if the task is invalid.
        if self._is_invalid_unit(unit_index):
            return lambda: None

        # 1. Locate the source memory.
        cpu_src = self._nvme_para_offload_buffer

        # 2. Offload the parameters.
        self._nvme_para_group.single_offload(
            FlexTrainDataID(Dtype.PARA, unit_index),
            cpu_src, index=1, async_op=True
        )

    def _submit_offload_opts(self, unit_index: int, forward: bool):
        """ Launch the async IO operation to offload NVMe optimizer states. """
        # If running in the auto-config mode, return.
        if self._auto_config:
            return
        # Return if the task is invalid.
        if self._is_invalid_unit(unit_index):
            return

        # 1. Locate the source memory.
        numel = self._nvme_fwd_opts_numel \
            if forward else self._nvme_bwd_offload_numel
        nvme_src = self._nvme_opts_offload_buffer[:numel]

        # 2. Launch the offload operation.
        self._nvme_opts_group.single_offload(
            FlexTrainDataID(Dtype.OPTS, unit_index),
            nvme_src, index=0 if forward else 1, async_op=True
        )

    def pre_micro_batch_forward(self, curr_task: LLMTask):
        """ Submit tasks for the given micro-batch in forward pass. """
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
        # Submit CUDA stream tasks to the post-recomputation function.
        transfer_grad = \
            self._submit_transfer_grad(unit_index + 1, micro_batch_index)

        def _pre_micro_batch_tasks():
            self._data_stream.submit(transfer_grad)
            self._data_stream.execute()
        set_post_recomputation_function(_pre_micro_batch_tasks)

    def post_micro_batch_forward(self, curr_task: LLMTask):
        """ Conduct post-processing after the forward of the micro-batch. """
        # If the optimizer is not initialized, return.
        if not self.is_initialized:
            return

        # Submit the optimizer step for the micro-batch.
        # NOTE: Due to PyTorch features, the CPU intensive operation below
        #       should be submitted after all CUDA operations.
        #       That's why it's here rather than in pre_micro_batch_forward.
        unit_index, micro_batch_index = curr_task.unit, curr_task.micro_batch
        self._execute_optimizer_step(unit_index + 2, micro_batch_index, True)

    def post_micro_batch_backward(self, curr_task: LLMTask):
        """ Conduct post-processing after the backward of the micro-batch. """
        # If the gradients are not compatible with the device dtype,
        # we need explicitly accumulate the gradients.
        if self._gradacc_dtype_incompatible:
            gradacc_buffer = self._gpu_bwd_receive_grads
            gradacc_buffer += self._gpu_bwd_extra_grads

        # Submit the optimizer step for the micro-batch.
        # Refer to post_micro_batch_forward for the reason why.
        unit_index, micro_batch_index = curr_task.unit, curr_task.micro_batch
        self._execute_optimizer_step(unit_index + 2, micro_batch_index, False)

    def pre_unit_forward(self, unit_index: int):
        # If the optimizer is not initialized, return.
        if not self.is_initialized:
            return

        # Rotate the buffers.
        self._rotate_buffers()

        self._submit_prefetch_opts(unit_index + 3, forward=True)
        self._plan_optimizer_steps(unit_index + 2, forward=True)
        self._submit_offload_opts(unit_index + 1, forward=True)

    def pre_unit_backward(self, unit_index: int):
        """ Prepare the unit for backward pass.

        Functions:
        1. Ensure the availability of the gradients buffer.

        Args:
            unit_index (int): Index of the unit.

        Returns:
            None
        """
        # Rotate the buffers.
        self._rotate_buffers()

        self._submit_prefetch_opts(unit_index + 1, forward=False)
        self._plan_optimizer_steps(unit_index + 2, forward=False)
        self._submit_offload_opts(unit_index + 3, forward=False)
        self._submit_offload_para(unit_index + 3)

        # Prepare the gradient buffer.
        self._prepare_unit_grads(unit_index)

    def _synchronize(self, rotate_buffers: bool = True):
        self._nvme_swapper.synchronize()
        self._data_stream.synchronize()
        if rotate_buffers:
            self._rotate_buffers()

    def warmup_forward_pipeline(self):
        # If the optimizer is not initialized, return.
        if not self.is_initialized:
            return

        # Load the first unit NVMe parameters to CPU.
        self._para._async_load_nvme_paras(0)

        # Synchronize the inflight tasks and rotate the buffers.
        self._synchronize()
        self._para._nvme_prefetch_buffer.rotate()

        # Conduct the pre-unit tasks for the first unit.
        self._submit_prefetch_opts(2, forward=True)
        self._submit_offload_opts(0, forward=True)
        self._plan_optimizer_steps(1, forward=True, execute_now=True)

        # Get the first unit parameters ready.
        # Prepare the second unit parameters.
        self._para._async_load_nvme_paras(1)
        self._para._async_offload_nvme_paras(0)
        for mb in range(self._micro_batch_per_rank):
            self._para._async_load_gpu_paras(0, mb)()

    def warmup_backward_pipeline(self):
        # Complete the initialization.
        self._initialize_alpha_split()
        # Inform the parameter coordinator that future updates are coming.
        self._para.parameter_updated = True

    def clear_backward_pipeline(self):
        """
        Cleanup the backward pipeline.
        """

        # Synchronize the inflight tasks and rotate the buffers.
        self._synchronize()
        # Conduct the last backward pre-unit task.
        self._submit_prefetch_opts(0, forward=False)
        self._submit_offload_opts(2, forward=False)
        self._submit_offload_para(2)
        # Conduct the last backward pre-micro-batch tasks.
        for mb in range(self._micro_batch_per_rank):
            self._submit_transfer_grad(0, mb)()
        self._plan_optimizer_steps(1, forward=False, execute_now=True)

        # Synchronize the inflight tasks and rotate the buffers.
        self._synchronize()
        # Conduct the last backward pre-unit task.
        self._submit_prefetch_opts(0, forward=True)
        self._submit_offload_opts(1, forward=False)
        self._submit_offload_para(1)
        self._plan_optimizer_steps(0, forward=False, execute_now=True)

        # Synchronize the inflight tasks and rotate the buffers.
        self._synchronize()
        # Conduct the last backward pre-unit task.
        self._submit_prefetch_opts(1, forward=True)
        self._submit_offload_opts(0, forward=False)
        self._submit_offload_para(0)
        self._plan_optimizer_steps(0, forward=True, execute_now=True)

        # Synchronize the inflight tasks.
        self._synchronize(rotate_buffers=False)


_OPTS_COORDINATOR = FlexTrainOptsCoordinator()


def get_opts_coordinator():
    """
    Get the optimizer coordinator.

    Returns:
        FlexTrainOptsCoordinator: The optimizer coordinator.
    """
    return _OPTS_COORDINATOR
