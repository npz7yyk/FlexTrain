import torch

from torch import Tensor
from torch.nn import Parameter
from typing import Iterable, List, Dict

from flextrain.config import get_flextrain_config
from flextrain.memory import (
    RotateContainer,
    FlexTrainDataType as Dtype,
    FlexTrainDataID,
    ContiguousParaGroup,
    get_partition_aligned_numel,
    get_page_aligned_padding_numel,
    get_split_numels,
    move_into_contiguous,
    allocate_memory_chunks,
    get_data_stream
)
from flextrain.memory.nvme_swapper import get_nvme_swapper, NVMeGroup
from flextrain.scheduler import LLMTask
from flextrain.utils import dist
from flextrain.utils.logging import rank0_logger


class FlexTrainParaCoordinator:

    def __init__(self):
        # Lazy initialization of parameter coordinator.
        self._initialized = False

    def _init_coordinator(
        self,
        num_layers: int,
        parameters: Iterable[Parameter]
    ):
        # If already initialized, return.
        if self._initialized:
            return
        self._initialized = True

        # assert CUDA is available
        assert torch.cuda.is_available(), \
            "FlexTrain requires CUDA to be available"

        # Init coordinator configurations.
        config = get_flextrain_config()

        # Mixed precision dtype for accelerator.
        self._device_dtype = config.mixed_precision.device_dtype
        # Mixed precision dtype for master.
        self._master_dtype = config.mixed_precision.master_dtype

        # Async IO NVMe swapper.
        self._nvme_swapper = get_nvme_swapper()
        # Async IO operation cuda stream.
        self._data_stream = get_data_stream()

        # Configuration for parameter partition.
        self._config_para_partition(parameters)

        # Number of layers in the model.
        self._num_layers = num_layers
        # Number of units in the model.
        self._num_units = (num_layers - 1) // config.checkpoint_interval + 1
        # Current unit index in model initialization.
        self._curr_unit = 0
        # Map of unit index to its parameters.
        self._unit_parameters: Dict[int, ContiguousParaGroup] = {}

        # Allocate parameter base containers.
        self._gpu_para_base = allocate_memory_chunks(
            self._micro_batch_para_splits[0],
            (self._num_units, self._micro_batch_per_rank),
            self._device_dtype, torch.cuda.current_device()
        )
        self._cpu_para_base = allocate_memory_chunks(
            self._micro_batch_para_splits[1],
            (self._num_units, self._micro_batch_per_rank),
            self._device_dtype, torch.device('cpu')
        )
        # Allocate GPU working memory for parameters.
        self._gpu_full_paras = RotateContainer(allocate_memory_chunks(
            self._aligned_unit_numel, 2,
            self._device_dtype, torch.cuda.current_device()
        ))
        # Allocate NVMe prefetch buffer in CPU memory.
        self._nvme_prefetch_buffer = RotateContainer(allocate_memory_chunks(
            self._nvme_group._group_numel, 2,
            self._device_dtype, torch.device('cpu')
        ))

        # Whether parameters are updated. Will be modified by optimizer.
        self.parameter_updated = False

    def _config_para_partition(self, parameters: Iterable[Parameter]):
        # Get the configuration.
        config = get_flextrain_config()
        assert config.split_ratio.parameter[0] == 0, (
            "GPU parameter support is dropped in FlexTrain, "
            "please set split_ratio.parameter[0] to 0."
        )
        assert config.split_ratio.alpha[0] <= 0.5, (
            f"FlexTrain only supports alpha <= 0.5, "
            f"but got {config.split_ratio.alpha[0]}."
        )

        # The original numel of the parameters in a unit.
        self._original_unit_numel = sum(p.numel() for p in parameters)

        # The number of micro-batches.
        num_micro_batches = config.batch_size // config.micro_batch_size
        self._micro_batch_per_rank = num_micro_batches // dist.get_world_size()
        self._num_micro_batches = num_micro_batches

        # The aligned numel of the parameters in a unit.
        self._aligned_unit_numel = get_partition_aligned_numel(
            self._original_unit_numel,
            self._num_micro_batches, self._device_dtype.itemsize
        )
        # The aligned numel of the parameters prepared in a micro-batch.
        assert self._aligned_unit_numel % self._num_micro_batches == 0
        self._aligned_micro_batch_numel = \
            self._aligned_unit_numel // self._num_micro_batches

        # How to split the parameters at micro-batch level.
        self._micro_batch_para_splits = get_split_numels(
            self._aligned_micro_batch_numel,
            config.split_ratio.parameter, self._device_dtype.itemsize
        )

        # How to split the GPU parameters at micro-batch level.
        self._micro_batch_gpu_alpha_splits = get_split_numels(
            self._micro_batch_para_splits[0],
            config.split_ratio.alpha, self._device_dtype.itemsize, num_levels=2
        )

        # How to split the CPU parameters at micro-batch level.
        self._micro_batch_cpu_alpha_splits = get_split_numels(
            self._micro_batch_para_splits[1],
            config.split_ratio.alpha, self._device_dtype.itemsize, num_levels=2
        )

        # How to split the NVMe parameters at micro-batch level.
        self._micro_batch_nvme_alpha_splits = get_split_numels(
            self._micro_batch_para_splits[2],
            config.split_ratio.alpha, self._device_dtype.itemsize, num_levels=2
        )
        # How to split the NVMe parameters at unit level.
        self._unit_nvme_alpha_splits = [
            split * self._micro_batch_per_rank
            for split in self._micro_batch_nvme_alpha_splits
        ]

        # The padding numel for NVMe parameters to be page-aligned.
        # Only the backward NVMe parameters need padding.
        self._unit_nvme_padding_numel = get_page_aligned_padding_numel(
            self._unit_nvme_alpha_splits[1], self._device_dtype.itemsize
        )
        # NVMe group for offloading and reloading a group of data.
        # Here it is used for offloading and reloading the NVMe parameters.
        self._nvme_group = NVMeGroup([
            self._unit_nvme_alpha_splits[0],
            self._unit_nvme_alpha_splits[1] + self._unit_nvme_padding_numel
        ])

    @property
    def is_initialized(self):
        return self._initialized

    @property
    def num_layers(self):
        return self._num_layers

    @property
    def num_units(self):
        return self._num_units

    @property
    def unit_numel(self):
        return self._aligned_unit_numel

    @property
    def unit_parameter_map(self):
        return self._unit_parameters

    @property
    def _gpu_inflight_paras(self):
        return self._gpu_full_paras[0]

    @property
    def _gpu_available_paras(self):
        return self._gpu_full_paras[1]

    @property
    def _nvme_inflight_paras(self):
        padding_numel = self._unit_nvme_padding_numel
        return self._nvme_prefetch_buffer[0][:-padding_numel]

    @property
    def _nvme_available_paras(self):
        padding_numel = self._unit_nvme_padding_numel
        return self._nvme_prefetch_buffer[1][:-padding_numel]

    @property
    def _nvme_asyncio_paras(self):
        return self._nvme_prefetch_buffer[0]

    @property
    def nvme_forward_update_paras(self):
        return torch.split(
            self._nvme_inflight_paras, self._unit_nvme_alpha_splits
        )[0]

    def _is_invalid_unit(self, unit_index: int):
        return unit_index < 0 or unit_index >= self._num_units

    def _is_invalid_micro_batch(self, micro_batch_index: int):
        return micro_batch_index < 0 or \
            micro_batch_index >= self._micro_batch_per_rank

    def _is_invalid_task(self, unit_index: int, micro_batch_index: int):
        return self._is_invalid_unit(unit_index) or \
            self._is_invalid_micro_batch(micro_batch_index)

    def _link_unit_parameters(
        self,
        unit_index: int,
        buffer: Tensor
    ):
        """
        Link the parameters in the unit to the buffer.
        We assume that data is reconstructed (or being) in the buffer.

        Args:
            unit_index (int): Index of the unit.
            buffer (Tensor): Buffer to link the parameters.

        Returns:
            None
        """
        # Check if the unit is valid.
        if self._is_invalid_unit(unit_index):
            return

        # Get the unit parameters.
        unit_paras = self._unit_parameters[unit_index]

        # Link the parameters.
        unit_paras.link_para_to(buffer)

    def _detach_unit_parameters(self, unit_index: int):
        """
        Detach the parameters in a unit from the memory.
        The parameters are no longer accessible.

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

        # Detach the parameters.
        unit_paras.detach_para()

    def init_unit_parameters(
        self,
        num_layers: int,
        unit_paras: Iterable[Parameter],
        param_grouping_mask: List[int]
    ):
        """
        Initialize the memory for the parameters in a unit.
        Parameters are moved into contiguous memory and partitioned.
        The contiguous memory is partitioned as:
        [(GPU, CPU, NVMe), (GPU, CPU, NVMe), ... (GPU, CPU, NVMe)].
        Each (GPU, CPU, NVMe) is corresponding to a micro-batch.

        Args:
            num_layers (int): Number of layers in the model.
            unit_parameters (List[Parameter]): List of parameters in a unit.
            param_grouping_mask (List[int]): Mask for grouping the parameters.

        Returns:
            None
        """

        # Initialize memory allocation if not done.
        self._init_coordinator(num_layers, unit_paras)

        # Track the unit parameters.
        self._unit_parameters[self._curr_unit] = ContiguousParaGroup(
            unit_paras, param_grouping_mask
        )

        # Using GPU working window to prepare the parameters.
        temp_gpu_buffer = self._gpu_inflight_paras
        # Using NVMe prefetch buffer to prepare the parameters.
        temp_nvme_buffer = self._nvme_inflight_paras

        # Move parameters into contiguous memory.
        move_into_contiguous(
            self._unit_parameters[self._curr_unit].paras, temp_gpu_buffer
        )

        # Partition parameters.
        micro_batch_partitioned_paras = torch.chunk(
            temp_gpu_buffer, self._micro_batch_per_rank
        )

        # Target memory buffer for each memory type.
        nvme_forward_tar, nvme_backward_tar = torch.split(
            temp_nvme_buffer, self._unit_nvme_alpha_splits
        )
        nvme_forward_partitions = torch.chunk(
            nvme_forward_tar, self._micro_batch_per_rank
        )
        nvme_backward_partitions = torch.chunk(
            nvme_backward_tar, self._micro_batch_per_rank
        )
        for mb, para in enumerate(micro_batch_partitioned_paras):
            # Locate the memory for each rank.
            tar_mem = torch.chunk(para, dist.get_world_size())[dist.get_rank()]
            gpu_view, cpu_view, nvme_view = torch.split(
                tar_mem, self._micro_batch_para_splits
            )
            # Store the views.
            self._gpu_para_base[self._curr_unit][mb].copy_(gpu_view)
            self._cpu_para_base[self._curr_unit][mb].copy_(cpu_view)
            # Split the NVMe view.
            nvme_forward_view, nvme_backward_view = torch.split(
                nvme_view, self._micro_batch_nvme_alpha_splits
            )
            nvme_forward_partitions[mb].copy_(nvme_forward_view)
            nvme_backward_partitions[mb].copy_(nvme_backward_view)

        # Offload the NVMe parameters.
        self._nvme_group.group_offload(
            FlexTrainDataID(Dtype.PARA, self._curr_unit),
            self._nvme_asyncio_paras
        )

        # Update the current unit index.
        self._curr_unit += 1

    def log_configuration(self):
        """
        Log the parameter coordinator configurations after initialization.
        """
        rank0_logger.info(
            "\n\n> "
            f"FlexTrain parameter coordinator initialized with configurations:"
            f"\n"
            f"  - number of workers: {dist.get_world_size()}\n"
            f"  - number of micro-batches per rank: "
            f"{self._micro_batch_per_rank}\n"
            f"  - device dtype: {self._device_dtype}\n"
            f"  - master dtype: {self._master_dtype}\n"
            f"  - number of layers: {self._num_layers}\n"
            f"  - number of units: {self._num_units}\n"
            f"  - numel per unit: {self._original_unit_numel}\n"
            f"  - numel per unit aligned: {self._aligned_unit_numel}\n"
            f"  - numel per micro-batch: {self._aligned_micro_batch_numel}\n"
            f"  - micro-batch split numels (GPU, CPU, NVMe): "
            f"{self._micro_batch_para_splits}\n"
        )

    def detach_all_parameters(self):
        """
        Detach all the parameters.
        """
        for unit_index in range(self._num_units):
            self._detach_unit_parameters(unit_index)

    def _async_load_nvme_paras(self, unit_index: int, forward=True):
        """
        Move NVMe parameters to CPU for further processing.
        The CPU parameters are prepared in _cpu_inflight_paras.

        Args:
            unit_index (int): Index of the unit.

        Returns:
            None
        """

        # Check if the unit is valid.
        if self._is_invalid_unit(unit_index):
            return

        # Register the inflight nvme loading unit.
        self._inflight_nvme_unit = unit_index

        if self.parameter_updated and forward:
            # If parameters are updated, only load backward NVMe parameters.
            fwd_numel = self._unit_nvme_alpha_splits[0]
            bwd_para = self._nvme_asyncio_paras[fwd_numel:]
            # Forward parameters are being copied from optimizer coordinator.
            # Only load backward NVMe parameters.
            self._nvme_group.single_reload(
                FlexTrainDataID(Dtype.PARA, unit_index),
                bwd_para, 1, async_op=True
            )
        else:
            # Load both forward and backward NVMe parameters.
            self._nvme_group.group_reload(
                FlexTrainDataID(Dtype.PARA, unit_index),
                self._nvme_asyncio_paras, async_op=True
            )

    def _async_offload_nvme_paras(self, unit_index: int):
        """
        Submit the offloading of updated forward parameters if needed.

        Args:
            unit_index (int): Index of the unit.

        Returns:
            None
        """
        # Check if the unit is valid.
        if self._is_invalid_unit(unit_index):
            return

        # If parameters are not updated, return.
        if not self.parameter_updated:
            return

        # Locate the forward NVMe parameters.
        fwd_para, _ = torch.split(
            self._nvme_available_paras, self._unit_nvme_alpha_splits
        )

        # Offload the forward NVMe parameters.
        self._nvme_group.single_offload(
            FlexTrainDataID(Dtype.PARA, unit_index),
            fwd_para, 0, async_op=True
        )

    def _validate_nvme_operations(self, unit_index: int):
        """
        Validate inflight NVMe load and potential offload operations.
        Ensure the NVMe parameters are ready in self._nvme_available_paras.

        Args:
            unit_index (int): Index of the unit.

        Returns:
            None
        """
        # Check if the unit is valid.
        if self._is_invalid_unit(unit_index):
            return

        # Validate the inflight NVMe load.
        assert self._inflight_nvme_unit == unit_index, (
            "Inflight NVMe load does not match the current task. "
            f"Expected unit index: {self._inflight_nvme_unit}, "
            f"actual unit index: {unit_index}."
        )

    def _async_load_gpu_paras(self, unit_index: int, micro_batch_index: int):
        """
        Prepare the GPU parameters for the unit.
        The GPU parameters are prepared in _gpu_inflight_paras.
        Task is submitted to the data stream (so not started yet).
        """
        # 0. Check if the unit is valid.
        if self._is_invalid_unit(unit_index):
            return lambda: None

        def load_gpu_para_task():
            # Basic memory partition.
            tar_full_paras = self._gpu_inflight_paras
            micro_batch_mem = torch.chunk(
                tar_full_paras, self._micro_batch_per_rank
            )[micro_batch_index]
            rank_mem = torch.chunk(
                micro_batch_mem, dist.get_world_size()
            )[dist.get_rank()]

            # Locate the destination memory.
            gpu_tar, cpu_tar, nvme_tar = torch.split(
                rank_mem, self._micro_batch_para_splits
            )
            nvme_forward_tar, nvme_backward_tar = torch.split(
                nvme_tar, self._micro_batch_nvme_alpha_splits
            )

            # Locate the source memory.
            gpu_src = self._gpu_para_base[unit_index][micro_batch_index]
            cpu_src = self._cpu_para_base[unit_index][micro_batch_index]
            nvme_forward, nvme_backward = torch.split(
                self._nvme_available_paras, self._unit_nvme_alpha_splits
            )
            nvme_forward_src = torch.chunk(
                nvme_forward, self._micro_batch_per_rank
            )[micro_batch_index]
            nvme_backward_src = torch.chunk(
                nvme_backward, self._micro_batch_per_rank
            )[micro_batch_index]

            # Copy parameters and potentially all-gather.
            gpu_tar.copy_(gpu_src, non_blocking=True)
            cpu_tar.copy_(cpu_src, non_blocking=True)
            nvme_forward_tar.copy_(nvme_forward_src, non_blocking=True)
            nvme_backward_tar.copy_(nvme_backward_src, non_blocking=True)
            dist.all_gather(micro_batch_mem, rank_mem, async_op=True)

        # 4. Return the task, will be submitted to the data stream.
        return load_gpu_para_task

    def pre_micro_batch_forward(self, curr_task: LLMTask):
        """
        Submit the prefetching task for the given micro-batch in forward pass.

        Args:
            curr_task (LLMTask): Current task.

        Returns:
            None
        """
        # Unpack the task.
        unit_index = curr_task.unit
        micro_batch_index = curr_task.micro_batch

        # Check if the unit and micro-batch are valid.
        if self._is_invalid_task(unit_index, micro_batch_index):
            return

        # Submit the preparation of parameters for the next unit.
        self._data_stream.submit(
            self._async_load_gpu_paras(unit_index + 1, micro_batch_index)
        )

    def pre_micro_batch_backward(self, curr_task: LLMTask):
        """
        Submit the prefetching task for the given micro-batch in backward pass.

        Args:
            curr_task (LLMTask): Current task.

        Returns:
            None
        """
        # Unpack the task.
        unit_index = curr_task.unit
        micro_batch_index = curr_task.micro_batch

        # Check if the unit and micro-batch are valid.
        if self._is_invalid_task(unit_index, micro_batch_index):
            return

        # Submit the preparation of parameters for the next unit.
        self._data_stream.submit(
            self._async_load_gpu_paras(unit_index - 1, micro_batch_index)
        )

    def pre_unit_forward(self, unit_index: int):
        """
        Prepare the unit for forward pass.

        Functions:
        1. Ensure the availability of the parameters.
        2. Kick off relevant prefetching tasks if needed.

        Args:
            unit_index (int): Index of the unit.

        Returns:
            None
        """
        # Check if the unit is valid.
        if self._is_invalid_unit(unit_index):
            return

        # Detach the parameters of the last unit.
        self._detach_unit_parameters(unit_index - 1)

        # Validate the inflight NVMe load and potential offload operations.
        self._validate_nvme_operations(unit_index + 1)

        # Just available buffer is now the prefetch buffer.
        # Just inflight buffer is now available after handle.wait().
        self._nvme_prefetch_buffer.rotate()
        # GPU parameters are ready in _gpu_inflight_paras.
        # Rotate the buffers to make them ready in _gpu_available_paras.
        self._gpu_full_paras.rotate()

        # Submit the prefetching of NVMe parameters.
        self._async_load_nvme_paras(unit_index + 2, forward=True)
        # Submit the offloading of NVMe parameters.
        self._async_offload_nvme_paras(unit_index + 1)

        # Link the parameters to the unit.
        self._link_unit_parameters(unit_index, self._gpu_available_paras)

    def pre_unit_backward(self, unit_index: int):
        """
        Prepare the unit for backward pass.

        Functions:
        1. Ensure the availability of the parameters.
        2. Kick off relevant prefetching tasks if needed.
        3. Allocate zeroed memory for gradients and link to the unit.

        Args:
            unit_index (int): Index of the unit.

        Returns:
            None
        """
        # Check if the unit is valid.
        if self._is_invalid_unit(unit_index):
            return

        # Detach the parameters of the last unit.
        self._detach_unit_parameters(unit_index + 1)

        # Validate the inflight NVMe load and potential offload operations.
        self._validate_nvme_operations(unit_index - 1)

        # Just available buffer is now the prefetch buffer.
        # Just inflight buffer is now available after handle.wait().
        self._nvme_prefetch_buffer.rotate()
        # GPU parameters are ready in _gpu_inflight_paras.
        # Rotate the buffers to make them ready in _gpu_available_paras.
        self._gpu_full_paras.rotate()

        # Submit the prefetching of NVMe parameters.
        self._async_load_nvme_paras(unit_index - 2, forward=False)

        # Link the parameters to the unit.
        self._link_unit_parameters(unit_index, self._gpu_available_paras)

    def warmup_forward_pipeline(self):
        # If parameters are updated, warm-up will be managed by optimizer.
        # Therefore, return directly.
        if self.parameter_updated:
            return

        # Load the first unit parameters to CPU.
        self._async_load_nvme_paras(0)
        self._nvme_swapper.synchronize()
        self._nvme_prefetch_buffer.rotate()

        # Launch the first unit forward.
        self._async_load_nvme_paras(1)
        for mb in range(self._micro_batch_per_rank):
            self._async_load_gpu_paras(0, mb)()
        self._data_stream.execute()

    def warmup_backward_pipeline(self):
        """
        Warm up the backward pipeline.
        """
        self._inflight_nvme_unit -= 1

    def clear_backward_pipeline(self):
        """
        Clear the backward pipeline.
        """
        self._detach_unit_parameters(0)


_PARA_COORDINATOR = FlexTrainParaCoordinator()


def get_para_coordinator():
    """
    Get the parameter coordinator.

    Returns:
        FlexTrainParaCoordinator: The parameter coordinator.
    """
    return _PARA_COORDINATOR
