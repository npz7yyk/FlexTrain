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
    DummyHandle,
    align_numel,
    get_split_numels,
    move_into_contiguous,
    allocate_memory_chunks,
    get_data_stream
)
from flextrain.memory.nvme_swapper import NVMeGroup
from flextrain.utils import dist
from flextrain.utils.logging import rank0_logger


class FlexTrainParaCoordinator:

    def __init__(self):
        # Lazy initialization of parameter coordinator.
        self._initialized = False

    def _init_coordinator(self, parameters: Iterable[Parameter]):
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

        # Lazy initialization of NVMe swapper.
        self._nvme_swapper = None
        # Async IO operation cuda stream.
        self._data_stream = get_data_stream()

        # Configuration for parameter partition.
        self._config_para_partition(parameters)

        # Number of layers in the model.
        self._num_layers = 0
        # Number of units in the model.
        self._num_units = 0
        # Map of unit index to its parameters.
        self._unit_parameters: Dict[int, ContiguousParaGroup] = {}

        # Allocate parameter base containers.
        # list[layer][micro_batch] -> Tensor
        self._gpu_para_base: List[List[Tensor]] = []
        self._cpu_para_base: List[List[Tensor]] = []
        # Allocate GPU working memory for parameters.
        self._gpu_full_paras = RotateContainer(allocate_memory_chunks(
            self._aligned_unit_numel, 2,
            self._device_dtype, torch.cuda.current_device()
        ))
        # Allocate NVMe prefetch buffer in CPU memory.
        self._nvme_prefetch_buffer = RotateContainer(allocate_memory_chunks(
            self._micro_batch_para_splits[2] * self._micro_batch_per_rank, 2,
            self._device_dtype, torch.device('cpu')
        ))

    def _config_para_partition(self, parameters: Iterable[Parameter]):
        # Get the configuration.
        config = get_flextrain_config()

        # The original numel of the parameters in a unit.
        self._original_unit_numel = sum(p.numel() for p in parameters)

        # The number of micro-batches.
        num_micro_batches = config.batch_size // config.micro_batch_size
        self._micro_batch_per_rank = num_micro_batches // dist.get_world_size()
        self._num_micro_batches = num_micro_batches

        # The aligned numel of the parameters in a unit.
        self._aligned_unit_numel = align_numel(
            self._original_unit_numel, self._num_micro_batches
        )
        # The aligned numel of the parameters prepared in a micro-batch.
        assert self._aligned_unit_numel % self._num_micro_batches == 0
        self._aligned_micro_batch_numel = \
            self._aligned_unit_numel // self._num_micro_batches

        # How to split the parameters at micro-batch level.
        self._micro_batch_para_splits = get_split_numels(
            self._aligned_micro_batch_numel, config.split_ratio.parameter
        )
        # Important!!!
        # We twist the order of the parameters to:
        # (GPU, CPU, NVMe) -> (CPU, GPU, NVMe)
        self._micro_batch_para_splits = (
            self._micro_batch_para_splits[1],
            self._micro_batch_para_splits[0],
            self._micro_batch_para_splits[2]
        )

        # How to split the CPU parameters at micro-batch level.
        self._micro_batch_cpu_alpha_splits = get_split_numels(
            self._micro_batch_para_splits[0],
            config.split_ratio.alpha, num_levels=2
        )
        # How to split the CPU parameters at unit level.
        self._unit_cpu_alpha_splits = [
            split * self._micro_batch_per_rank
            for split in self._micro_batch_cpu_alpha_splits
        ]

        # How to split the GPU parameters at micro-batch level.
        self._micro_batch_gpu_alpha_splits = get_split_numels(
            self._micro_batch_para_splits[1],
            config.split_ratio.alpha, num_levels=2
        )
        # How to split the GPU parameters at unit level.
        self._unit_gpu_alpha_splits = [
            split * self._micro_batch_per_rank
            for split in self._micro_batch_gpu_alpha_splits
        ]

        # How to split the NVMe parameters at micro-batch level.
        self._micro_batch_nvme_alpha_splits = get_split_numels(
            self._micro_batch_para_splits[2],
            config.split_ratio.alpha, num_levels=2
        )
        # How to split the NVMe parameters at unit level.
        self._unit_nvme_alpha_splits = [
            split * self._micro_batch_per_rank
            for split in self._micro_batch_nvme_alpha_splits
        ]

        # NVMe group for offloading and reloading a group of data.
        # Here it is used for offloading and reloading the NVMe parameters.
        self._nvme_group = NVMeGroup(self._unit_nvme_alpha_splits)

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
        return self._nvme_prefetch_buffer[0]

    @property
    def _nvme_available_paras(self):
        return self._nvme_prefetch_buffer[1]

    def _is_invalid_unit(self, unit_index: int):
        return unit_index < 0 or unit_index >= self._num_units

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
        unit_index: int,
        unit_paras: Iterable[Parameter]
    ):
        """
        Initialize the memory for the parameters in a unit.
        Parameters are moved into contiguous memory and partitioned.
        The contiguous memory is partitioned as:
        [(GPU, CPU, NVMe), (GPU, CPU, NVMe), ... (GPU, CPU, NVMe)].
        Each (GPU, CPU, NVMe) is corresponding to a micro-batch.

        Args:
            num_layers (int): Number of layers in the unit.
            unit_index (int): Index of the unit.
            unit_parameters (List[Parameter]): \
                List of parameters in a unit.

        Returns:
            None
        """

        # Initialize memory allocation if not done.
        self._init_coordinator(unit_paras)

        # Update the number of layers.
        self._num_layers += num_layers
        # Update the number of units.
        self._num_units += 1

        # Allocate parameter bases
        self._cpu_para_base.append(allocate_memory_chunks(
            self._micro_batch_para_splits[0], self._micro_batch_per_rank,
            self._device_dtype, torch.device('cpu')
        ))
        self._gpu_para_base.append(allocate_memory_chunks(
            self._micro_batch_para_splits[1], self._micro_batch_per_rank,
            self._device_dtype, torch.cuda.current_device()
        ))

        # Track the unit parameters.
        self._unit_parameters[unit_index] = ContiguousParaGroup(unit_paras)

        # Using GPU working window to prepare the parameters.
        temp_gpu_buffer = self._gpu_inflight_paras

        # Move parameters into contiguous memory.
        move_into_contiguous(unit_paras, temp_gpu_buffer)

        # Partition parameters.
        micro_batch_partitioned_paras = torch.chunk(
            temp_gpu_buffer, self._micro_batch_per_rank
        )

        # Target memory buffer for each memory type.
        nvme_forward_tar, nvme_backward_tar = torch.split(
            self._nvme_available_paras, self._unit_nvme_alpha_splits
        )
        nvme_forward_partitions = torch.chunk(
            nvme_forward_tar, self._micro_batch_per_rank
        )
        nvme_backward_partitions = torch.chunk(
            nvme_backward_tar, self._micro_batch_per_rank
        )
        for i, para in enumerate(micro_batch_partitioned_paras):
            # Locate the memory for each rank.
            tar_mem = torch.chunk(para, dist.get_world_size())[dist.get_rank()]
            cpu_view, gpu_view, nvme_view = torch.split(
                tar_mem, self._micro_batch_para_splits
            )
            # Store the views.
            self._gpu_para_base[unit_index][i].copy_(gpu_view)
            self._cpu_para_base[unit_index][i].copy_(cpu_view)

            # Split the NVMe view.
            nvme_forward_view, nvme_backward_view = torch.split(
                nvme_view, self._micro_batch_nvme_alpha_splits
            )
            nvme_forward_partitions[i].copy_(nvme_forward_view)
            nvme_backward_partitions[i].copy_(nvme_backward_view)

        # Offload the NVMe parameters.
        self._nvme_group.group_offload(
            FlexTrainDataID(unit_index, Dtype.PARA), self._nvme_available_paras
        )

        # Detach the parameters from the memory.
        self._detach_unit_parameters(unit_index)

    def log_configuration(self):
        """
        Log the parameter coordinator configurations after initialization.
        """
        rank0_logger.info(
            "\n\n> "
            f"FlexTrain parameter coordinator initialized with configurations:"
            f"\n"
            f"  - number of ranks: {dist.get_world_size()}\n"
            f"  - number of micro-batches per rank: "
            f"{self._micro_batch_per_rank}\n"
            f"  - device dtype: {self._device_dtype}\n"
            f"  - master dtype: {self._master_dtype}\n"
            f"  - number of layers: {self._num_layers}\n"
            f"  - number of units: {self._num_units}\n"
            f"  - numel per unit: {self._original_unit_numel}\n"
            f"  - numel per unit aligned: {self._aligned_unit_numel}\n"
            f"  - numel per micro-batch: {self._aligned_micro_batch_numel}\n"
            f"  - micro-batch split numels (CPU, GPU, NVMe): "
            f"{self._micro_batch_para_splits}\n"
        )

    def _async_load_cpu_paras(self, unit_index: int):
        """
        Move NVMe parameters to CPU for further processing.
        The CPU parameters are prepared in _cpu_inflight_paras.

        Args:
            unit_index (int): Index of the unit.

        Returns:
            Waitable: Handle for the async IO operation.
        """
        if self._is_invalid_unit(unit_index):
            return DummyHandle()

        return self._nvme_group.group_reload(
            FlexTrainDataID(unit_index, Dtype.PARA),
            self._nvme_inflight_paras, async_op=True
        )

    def _async_load_gpu_paras(self, unit_index: int, micro_batch_index: int):
        """
        Prepare the GPU parameters for the unit.
        The GPU parameters are prepared in _gpu_inflight_paras.
        Task is submitted to the data stream (so not started yet).
        """
        # 0. Check if the unit is valid.
        if self._is_invalid_unit(unit_index):
            return

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
            cpu_tar, gpu_tar, nvme_tar = torch.split(
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

        # Submit the task to the data stream.
        self._data_stream.submit(load_gpu_para_task)

    def _submit_prepare_paras(self, unit_index: int, forward: bool = True):
        """
        Launch the async IO operation to prepare parameters for the unit.

        Args:
            unit_index (int): Index of the unit.

        Returns:
            None
        """
        # Check if the unit is valid.
        if self._is_invalid_unit(unit_index):
            return

        # Prepare the CPU parameters.
        offset = 1 if forward else -1
        cpu_handle = self._async_load_cpu_paras(unit_index + offset)

        # Keep track of the inflight operation.
        self._inflight_para_handle = (unit_index, cpu_handle)

    def _synchronize_prepare_paras(self, unit_index: int):
        """
        Synchronize the preparation of parameters for given unit.
        Ensure that parameters are ready in the _gpu_available_paras.

        Args:
            unit_index (int): Index of the unit.

        Returns:
            None
        """
        # Check if the unit is valid.
        if self._is_invalid_unit(unit_index):
            return

        # Wait for the async IO operation to finish.
        inflight_unit, handle = self._inflight_para_handle
        assert inflight_unit == unit_index, (
            f"Async IO operation is not for this unit: "
            f"unit_index={unit_index} != inflight_unit={inflight_unit}"
        )
        handle.wait()

        # Just available buffer is now the prefetch buffer.
        # Just inflight buffer is now available after handle.wait().
        self._nvme_prefetch_buffer.rotate()

        # Just inflight GPU buffer is now available for unit forward.
        # Just available GPU buffer is now free for prefetching.
        self._gpu_full_paras.rotate()

        # Link the parameters to the available buffer.
        self._link_unit_parameters(unit_index, self._gpu_available_paras)

    def warmup_forward_pipeline(self):
        """
        Warm up the forward pipeline.
        Recommendation: call before LLM pre_processing for further speedup.
        """

        # Load the first unit parameters to CPU.
        self._async_load_cpu_paras(0).wait()
        self._nvme_prefetch_buffer.rotate()

        # Launch the first unit forward.
        self._submit_prepare_paras(0)
        for i in range(self._micro_batch_per_rank):
            self._async_load_gpu_paras(0, i)
        self._data_stream.execute()

    def pre_micro_batch_forward(self, unit_index: int, micro_batch_index: int):
        """
        Submit the prefetching task for the given micro-batch in forward pass.

        Args:
            unit_index (int): Index of the unit.
            micro_batch_index (int): Index of the micro-batch.

        Returns:
            None
        """
        # Check if the unit is valid.
        if self._is_invalid_unit(unit_index):
            return

        self._async_load_gpu_paras(unit_index + 1, micro_batch_index)

    def pre_micro_batch_backward(
        self, unit_index: int, micro_batch_index: int
    ):
        """
        Submit the prefetching task for the given micro-batch in backward pass.

        Args:
            unit_index (int): Index of the unit.
            micro_batch_index (int): Index of the micro-batch.

        Returns:
            None
        """
        # Check if the unit is valid.
        if self._is_invalid_unit(unit_index):
            return

        self._async_load_gpu_paras(unit_index - 1, micro_batch_index)

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

        self._detach_unit_parameters(unit_index - 1)
        self._synchronize_prepare_paras(unit_index)
        self._submit_prepare_paras(unit_index + 1)

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

        self._detach_unit_parameters(unit_index + 1)
        self._synchronize_prepare_paras(unit_index)
        self._submit_prepare_paras(unit_index - 1, forward=False)

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
