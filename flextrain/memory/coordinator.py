import torch

from dataclasses import dataclass
from math import ceil
from torch import Tensor
from torch.nn import Parameter
from typing import SupportsIndex, Iterable, Callable, Tuple, List, Dict

from flextrain.config import get_flextrain_config
from flextrain.memory import (
    free_tensor,
    FlexTrainDataType as Dtype,
    FlexTrainDataID,
    align_numel,
    move_into_contiguous,
    ContiguousParaGroup,
    Waitable,
    DummyHandle,
    FunctionHandle
)
from flextrain.memory.nvme_swapper import AsyncNVMeSwapper
from flextrain.utils import dist
from flextrain.utils.logging import rank0_logger


class RotateContainer:
    def __init__(self, items: Tuple):
        self._items = list(items)

    def __getitem__(self, index: SupportsIndex):
        return self._items[index]

    def rotate(self):
        self._items.append(self._items.pop(0))


def _allocate_memory_chunks(
    numel: int,
    chunks: int | Tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device
):
    # Wrap the chunks into a tuple.
    if isinstance(chunks, int):
        chunks = (chunks,)

    # Calculate the total memory size.
    total_numel = numel
    for dim in chunks:
        total_numel *= dim

    device = torch.device(device)
    return torch.empty(
        total_numel, dtype=dtype, device=device,
        pin_memory=True if device.type == 'cpu' else False
    ).reshape(*chunks, numel)


def _filter_tensors(
    data_ids: FlexTrainDataID | Iterable[FlexTrainDataID],
    tensors: Tensor | Iterable[Tensor]
):
    if isinstance(data_ids, FlexTrainDataID):
        data_ids = [data_ids]
    if isinstance(tensors, Tensor):
        tensors = [tensors]

    non_empty_data_ids = []
    non_empty_tensors = []
    for data_id, tensor in zip(data_ids, tensors):
        if tensor.numel() == 0:
            continue
        non_empty_data_ids.append(data_id)
        non_empty_tensors.append(tensor)

    return non_empty_data_ids, non_empty_tensors


def _get_split_numels(
    total_numel: int,
    ratios: Iterable[float],
    num_levels: int = 3
):
    # Ensure the number of levels is 2.
    if len(ratios) == num_levels:
        ratios = ratios[:num_levels - 1]

    # User provides integer splits, compute the rest.
    if sum(ratios) > 1 and all(isinstance(r, int) for r in ratios):
        numels = ratios + [total_numel - sum(ratios)]
        return tuple(numels)

    # Try to avoid the last one being 0.
    numels = [ceil(r * total_numel) for r in ratios]
    if sum(numels) > total_numel:
        numels[-1] -= sum(numels) - total_numel
    numels.append(total_numel - sum(numels))
    return tuple(numels)


_NVME_SWAPPER: AsyncNVMeSwapper = None


def _nvme_offload(
    data_ids: FlexTrainDataID | Iterable[FlexTrainDataID],
    tensors: Tensor | Iterable[Tensor],
    async_op: bool = False
):
    # Filter out empty tensors.
    data_ids, tensors = _filter_tensors(data_ids, tensors)

    # If no tensors to offload, return.
    if len(data_ids) == 0:
        return DummyHandle() if async_op else None

    # Lazy initialization of the NVMe swapper.
    global _NVME_SWAPPER
    if _NVME_SWAPPER is None:
        nvme_swap_config = get_flextrain_config().nvme_swap
        _NVME_SWAPPER = AsyncNVMeSwapper(
            swap_dir=nvme_swap_config.swap_dir,
            aio_block_size=nvme_swap_config.aio_block_size,
            aio_queue_depth=nvme_swap_config.aio_queue_depth,
            aio_thread_count=nvme_swap_config.aio_thread_count,
            aio_single_submit=nvme_swap_config.aio_single_submit,
            aio_overlap_events=nvme_swap_config.aio_overlap_events
        )

    # Call the NVMe swapper.
    return _NVME_SWAPPER.swap_out(data_ids, tensors, async_op)


def _nvme_reload(
    data_ids: FlexTrainDataID | Iterable[FlexTrainDataID],
    tensors: Tensor | Iterable[Tensor],
    async_op: bool = False
):
    # If swapper is not initialized, return.
    if _NVME_SWAPPER is None:
        return DummyHandle() if async_op else None

    # Filter out empty tensors.
    data_ids, tensors = _filter_tensors(data_ids, tensors)

    # If no tensors to reload, return.
    if len(data_ids) == 0:
        return DummyHandle() if async_op else None

    # Call the NVMe swapper.
    return _NVME_SWAPPER.swap_in(data_ids, tensors, async_op)


class FlexTrainDataStream:

    def __init__(self):
        self._stream = torch.cuda.Stream()
        self._tasks: List[Callable] = []

    def is_empty(self):
        return len(self._tasks) == 0

    def submit(self, task):
        self._tasks.append(task)

    def execute(self):
        with torch.cuda.stream(self._stream):
            for task in self._tasks:
                task()
        self._tasks.clear()

    def synchronize(self):
        torch.cuda.synchronize()


_DATA_STREAM: FlexTrainDataStream = None


def get_data_stream():
    """
    Get the data stream for async IO operations.

    Returns:
        FlexTrainDataStream: The data stream.
    """
    # Lazy initialization of the data stream
    global _DATA_STREAM
    if _DATA_STREAM is None:
        _DATA_STREAM = FlexTrainDataStream()
    return _DATA_STREAM


class NVMeGroup:

    def __init__(self, numels: Tuple[int, ...]):
        self._numels = numels
        self._group_numel = sum(numels)

    def _rename(self, prefix: FlexTrainDataID, index: int):
        return str(prefix) + f".{index}"

    def single_offload(
        self,
        prefix: FlexTrainDataID,
        tensor: Tensor,
        index: int,
        async_op: bool = False
    ) -> Waitable:
        assert tensor.numel() == self._numels[index]
        return _nvme_offload(self._rename(prefix, index), tensor, async_op)

    def group_offload(
        self,
        prefix: FlexTrainDataID,
        tensors: Tensor | Iterable[Tensor],
        async_op: bool = False
    ) -> Waitable:
        if isinstance(tensors, Tensor):
            assert tensors.numel() == self._group_numel, (
                f"Expected numel={self._group_numel}, "
                f"got numel={tensors.numel()}"
            )
            tensors = torch.split(tensors, self._numels)

        for tensor, numel in zip(tensors, self._numels):
            assert tensor.numel() == numel

        return _nvme_offload(
            [self._rename(prefix, i) for i in range(len(tensors))],
            tensors, async_op
        )

    def single_reload(
        self,
        prefix: FlexTrainDataID,
        tensor: Tensor,
        index: int,
        async_op: bool = False
    ) -> Waitable:
        assert tensor.numel() == self._numels[index]
        return _nvme_reload(self._rename(prefix, index), tensor, async_op)

    def group_reload(
        self,
        prefix: FlexTrainDataID,
        tensors: Tensor | Iterable[Tensor],
        async_op: bool = False
    ) -> Waitable:
        if isinstance(tensors, Tensor):
            assert tensors.numel() == self._group_numel
            tensors = torch.split(tensors, self._numels)

        for tensor, numel in zip(tensors, self._numels):
            assert tensor.numel() == numel

        return _nvme_reload(
            [self._rename(prefix, i) for i in range(len(tensors))],
            tensors, async_op
        )


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
        self._gpu_full_paras = RotateContainer(_allocate_memory_chunks(
            self._aligned_unit_numel, 2,
            self._device_dtype, torch.cuda.current_device()
        ))
        # Allocate NVMe prefetch buffer in CPU memory.
        self._nvme_prefetch_buffer = RotateContainer(_allocate_memory_chunks(
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
        self._micro_batch_para_splits = _get_split_numels(
            self._aligned_micro_batch_numel, config.split_ratio.parameter
        )

        # How to split the GPU parameters at micro-batch level.
        self._micro_batch_gpu_alpha_splits = _get_split_numels(
            self._micro_batch_para_splits[0],
            config.split_ratio.alpha, num_levels=2
        )
        # How to split the GPU parameters at unit level.
        self._unit_gpu_alpha_splits = [
            split * self._micro_batch_per_rank
            for split in self._micro_batch_gpu_alpha_splits
        ]

        # How to split the CPU parameters at micro-batch level.
        self._micro_batch_cpu_alpha_splits = _get_split_numels(
            self._micro_batch_para_splits[1],
            config.split_ratio.alpha, num_levels=2
        )
        # How to split the CPU parameters at unit level.
        self._unit_cpu_alpha_splits = [
            split * self._micro_batch_per_rank
            for split in self._micro_batch_cpu_alpha_splits
        ]

        # How to split the NVMe parameters at micro-batch level.
        self._micro_batch_nvme_alpha_splits = _get_split_numels(
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
        self._gpu_para_base.append(_allocate_memory_chunks(
            self._micro_batch_para_splits[0], self._micro_batch_per_rank,
            self._device_dtype, torch.cuda.current_device()
        ))
        self._cpu_para_base.append(_allocate_memory_chunks(
            self._micro_batch_para_splits[1], self._micro_batch_per_rank,
            self._device_dtype, torch.device('cpu')
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
        nvme_backward_tar, nvme_forward_tar = torch.split(
            self._nvme_available_paras, self._unit_nvme_alpha_splits
        )
        nvme_backward_partitions = torch.chunk(
            nvme_backward_tar, self._micro_batch_per_rank
        )
        nvme_forward_partitions = torch.chunk(
            nvme_forward_tar, self._micro_batch_per_rank
        )
        for i, para in enumerate(micro_batch_partitioned_paras):
            # Locate the memory for each rank.
            tar_mem = torch.chunk(para, dist.get_world_size())[dist.get_rank()]
            gpu_view, cpu_view, nvme_view = torch.split(
                tar_mem, self._micro_batch_para_splits
            )
            # Store the views.
            self._gpu_para_base[unit_index][i].copy_(gpu_view)
            self._cpu_para_base[unit_index][i].copy_(cpu_view)

            # Split the NVMe view.
            nvme_backward_view, nvme_forward_view = torch.split(
                nvme_view, self._micro_batch_nvme_alpha_splits
            )
            nvme_backward_partitions[i].copy_(nvme_backward_view)
            nvme_forward_partitions[i].copy_(nvme_forward_view)

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
            f"  - micro-batch split numels: {self._micro_batch_para_splits}\n"
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
            gpu_tar, cpu_tar, nvme_tar = torch.split(
                rank_mem, self._micro_batch_para_splits
            )
            nvme_backward_tar, nvme_forward_tar = torch.split(
                nvme_tar, self._micro_batch_nvme_alpha_splits
            )

            # Locate the source memory.
            gpu_src = self._gpu_para_base[unit_index][micro_batch_index]
            cpu_src = self._cpu_para_base[unit_index][micro_batch_index]
            nvme_backward, nvme_forward = torch.split(
                self._nvme_available_paras, self._unit_nvme_alpha_splits
            )
            nvme_backward_src = torch.chunk(
                nvme_backward, self._micro_batch_per_rank
            )[micro_batch_index]
            nvme_forward_src = torch.chunk(
                nvme_forward, self._micro_batch_per_rank
            )[micro_batch_index]

            # Copy parameters and potentially all-gather.
            gpu_tar.copy_(gpu_src, non_blocking=True)
            cpu_tar.copy_(cpu_src, non_blocking=True)
            nvme_backward_tar.copy_(nvme_backward_src, non_blocking=True)
            nvme_forward_tar.copy_(nvme_forward_src, non_blocking=True)
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

        self._synchronize_prepare_paras(unit_index)
        self._submit_prepare_paras(unit_index - 1, forward=False)


_PARA_COORDINATOR = FlexTrainParaCoordinator()


def get_para_coordinator():
    """
    Get the parameter coordinator.

    Returns:
        FlexTrainParaCoordinator: The parameter coordinator.
    """
    return _PARA_COORDINATOR


@dataclass
class InterLayerTask:
    unit: int
    micro_batch: int
    tensor: Tensor = None


def retrieve_tensor(interlayer: Tensor | Tuple[Tensor, ...]) -> Tensor:
    # If interlayer is a single tensor, return it.
    if isinstance(interlayer, Tensor):
        return interlayer

    # Unpack the tuple.
    # Currently, only one tensor is supported.
    tar = None
    for tensor in interlayer:
        if isinstance(tensor, Tensor):
            assert tar is None, (
                "Currently, only one tensor is supported for FlexTrain "
                "checkpointing. You may consider manually place all "
                "inter-layer results into a single tensor."
            )
            tar = tensor

    assert tar is not None, "No tensor can be found in inter-layer results."
    return tar


class FlexTrainInterLayerCoordinator:

    def __init__(self):
        # Lazy initialization of checkpoint coordinator.
        self._initialized = False

    @property
    def is_initialized(self):
        return self._initialized

    def _init_coordinator(self, tensor: Tensor):
        # If already initialized, return.
        if self._initialized:
            return
        self._initialized = True

        # Memory lazy allocation.
        self._mem_allocated = False

        # Record the tensor shape.
        self._tensor_shape = tensor.shape

        # CUDA streams for async IO operations.
        self._data_stream = get_data_stream()
        # Last task handle.
        self._inflight_handle: Waitable = DummyHandle()

        # Initialize coordinator configurations.
        config = get_flextrain_config()
        self._num_units = get_para_coordinator().num_units
        num_micro_batches = config.batch_size // config.micro_batch_size
        self._micro_batch_per_rank = num_micro_batches // dist.get_world_size()

        # How to split the checkpoint tensor.
        self._ckpt_numels = _get_split_numels(
            tensor.numel(), config.split_ratio.checkpoint, num_levels=2
        )
        # How to split the gradient tensor.
        self._grad_numels = _get_split_numels(
            tensor.numel(), config.split_ratio.gradient, num_levels=2
        )

        # Allocate memory for checkpoint.
        self.gpu_ckpt_base = _allocate_memory_chunks(
            self._ckpt_numels[0],
            (self._num_units - 1, self._micro_batch_per_rank),
            config.mixed_precision.device_dtype,
            torch.cuda.current_device()
        )
        self.cpu_ckpt_base = _allocate_memory_chunks(
            self._ckpt_numels[1],
            (self._num_units - 1, self._micro_batch_per_rank),
            config.mixed_precision.device_dtype,
            torch.device('cpu')
        )

        # Allocate memory for gradient.
        self.gpu_grad_base = _allocate_memory_chunks(
            self._grad_numels[0],
            self._micro_batch_per_rank,
            config.mixed_precision.device_dtype,
            torch.cuda.current_device()
        )
        self.cpu_grad_base = _allocate_memory_chunks(
            self._grad_numels[1],
            self._micro_batch_per_rank,
            config.mixed_precision.device_dtype,
            torch.device('cpu')
        )

        # Allocate memory for GPU checkpoint buffer.
        self._gpu_full_ckpts = RotateContainer(
            _allocate_memory_chunks(
                tensor.numel(), 2,
                config.mixed_precision.device_dtype,
                torch.cuda.current_device()
            )
        )
        # Allocate memory for GPU gradient buffer.
        self._gpu_full_grads = RotateContainer(
            _allocate_memory_chunks(
                tensor.numel(), 2,
                config.mixed_precision.device_dtype,
                torch.cuda.current_device()
            )
        )

        # Log the configuration.
        rank0_logger.info(
            "\n\n> "
            f"FlexTrain inter-layer coordinator initialized "
            f"with configurations:\n"
            f"  - checkpoint split numels (GPU, CPU): {self._ckpt_numels}\n"
            f"  - gradient split numels (GPU, CPU): {self._grad_numels}\n"
        )

    def _mask_invalid_task(self, task: InterLayerTask):
        if not self._initialized or task is None:
            return task
        unit = task.unit
        micro_batch = task.micro_batch
        if unit < 0 or unit >= self._num_units - 1:
            return None
        elif micro_batch < 0 or micro_batch >= self._micro_batch_per_rank:
            return None
        else:
            return task

    @property
    def inflight_layer_ckpt(self):
        return self._gpu_full_ckpts[0]

    @property
    def available_layer_ckpt(self):
        ckpt_mem: Tensor = self._gpu_full_ckpts[1]
        return ckpt_mem.view(self._tensor_shape)

    @property
    def inflight_layer_grad(self):
        return self._gpu_full_grads[0]

    @property
    def available_layer_grad(self):
        grad_mem: Tensor = self._gpu_full_grads[1]
        return grad_mem.view(self._tensor_shape)

    def _sync_pre_micro_batch_forward(self):
        self._inflight_handle.wait()
        self._gpu_full_ckpts.rotate()

    def _prefetch_ckpt(self, task: InterLayerTask):
        tar = self.inflight_layer_ckpt
        gpu_tar, cpu_tar = tar.split(self._ckpt_numels)
        gpu_src = self.gpu_ckpt_base[task.unit][task.micro_batch]
        cpu_src = self.cpu_ckpt_base[task.unit][task.micro_batch]
        gpu_tar.copy_(gpu_src, non_blocking=True)
        cpu_tar.copy_(cpu_src, non_blocking=True)

    def _offload_ckpt(self, task: InterLayerTask):
        src = task.tensor.flatten()
        gpu_src, cpu_src = src.split(self._ckpt_numels)
        gpu_tar = self.gpu_ckpt_base[task.unit][task.micro_batch]
        cpu_tar = self.cpu_ckpt_base[task.unit][task.micro_batch]
        gpu_tar.copy_(gpu_src, non_blocking=True)
        cpu_tar.copy_(cpu_src, non_blocking=True)

    def _submit_pre_micro_batch_forward(
        self,
        ckpt_prefetch: InterLayerTask = None,
        ckpt_offload: InterLayerTask = None
    ):
        def interlayer_task():
            if ckpt_prefetch is not None:
                self._prefetch_ckpt(ckpt_prefetch)
            if ckpt_offload is not None:
                self._offload_ckpt(ckpt_offload)

        self._data_stream.submit(interlayer_task)

        def free_ckpt_memory():
            if ckpt_offload is not None and ckpt_offload.tensor is not None:
                free_tensor(ckpt_offload.tensor)

        self._inflight_handle = FunctionHandle(free_ckpt_memory)

    def pre_micro_batch_forward(
        self,
        ckpt_prefetch: InterLayerTask = None,
        ckpt_offload: InterLayerTask = None
    ):
        # Mask invalid tasks.
        ckpt_prefetch = self._mask_invalid_task(ckpt_prefetch)
        ckpt_offload = self._mask_invalid_task(ckpt_offload)

        # Initialize coordinator if not done.
        if not self._initialized:
            if ckpt_offload is not None:
                self._init_coordinator(ckpt_offload.tensor)

        # If still not initialized, return.
        if not self._initialized:
            return

        self._sync_pre_micro_batch_forward()
        self._submit_pre_micro_batch_forward(ckpt_prefetch, ckpt_offload)

    def _sync_pre_micro_batch_backward(self):
        self._inflight_handle.wait()
        self._gpu_full_ckpts.rotate()
        self._gpu_full_grads.rotate()

    def _grad_prefetch(self, task: InterLayerTask):
        tar = self.inflight_layer_grad
        gpu_tar, cpu_tar = tar.split(self._grad_numels)
        gpu_src = self.gpu_grad_base[task.micro_batch]
        cpu_src = self.cpu_grad_base[task.micro_batch]
        gpu_tar.copy_(gpu_src, non_blocking=True)
        cpu_tar.copy_(cpu_src, non_blocking=True)

    def _grad_offload(self, task: InterLayerTask):
        src = task.tensor.flatten()
        gpu_src, cpu_src = src.split(self._grad_numels)
        gpu_tar = self.gpu_grad_base[task.micro_batch]
        cpu_tar = self.cpu_grad_base[task.micro_batch]
        gpu_tar.copy_(gpu_src, non_blocking=True)
        cpu_tar.copy_(cpu_src, non_blocking=True)

    def _submit_pre_micro_batch_backward(
        self,
        ckpt_prefetch: InterLayerTask = None,
        grad_prefetch: InterLayerTask = None,
        grad_offload: InterLayerTask = None
    ):
        def interlayer_task():
            if ckpt_prefetch is not None:
                self._prefetch_ckpt(ckpt_prefetch)
            if grad_prefetch is not None:
                self._grad_prefetch(grad_prefetch)
            if grad_offload is not None:
                self._grad_offload(grad_offload)

        self._data_stream.submit(interlayer_task)

        def free_grad_memory():
            if grad_offload is not None and grad_offload.tensor is not None:
                free_tensor(grad_offload.tensor)

        self._inflight_handle = FunctionHandle(free_grad_memory)

    def pre_micro_batch_backward(
        self,
        ckpt_prefetch: InterLayerTask = None,
        grad_prefetch: InterLayerTask = None,
        grad_offload: InterLayerTask = None
    ):
        # Mask invalid tasks.
        ckpt_prefetch = self._mask_invalid_task(ckpt_prefetch)
        grad_prefetch = self._mask_invalid_task(grad_prefetch)
        grad_offload = self._mask_invalid_task(grad_offload)

        self._sync_pre_micro_batch_backward()
        self._submit_pre_micro_batch_backward(
            ckpt_prefetch, grad_prefetch, grad_offload
        )


_INTERLAYER_COORDINATOR = FlexTrainInterLayerCoordinator()


def get_interlayer_coordinator():
    """
    Get the inter-layer coordinator.

    Returns:
        FlexTrainInterLayerCoordinator: The inter-layer coordinator.
    """
    return _INTERLAYER_COORDINATOR


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

    def initialize(self):
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
        self._ckpt_bound = False

        # 1. Set the configuration for the optimizer.
        self._num_units = para.num_units
        self._unit_parameters = para._unit_parameters

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
        self._gradacc_dtype_incompatible = device_dtype != gradacc_dtype

        # Configuration for optimizer partition.
        self._unit_numel = para._aligned_unit_numel
        self._micro_batch_per_rank = para._micro_batch_per_rank

        self._mb_gpu_para_alpha_splits = para._micro_batch_gpu_alpha_splits
        self._mb_cpu_para_alpha_splits = para._micro_batch_cpu_alpha_splits
        self._mb_nvme_para_alpha_splits = para._micro_batch_nvme_alpha_splits
        self._unit_gpu_para_alpha_splits = para._unit_gpu_alpha_splits
        self._unit_cpu_para_alpha_splits = para._unit_cpu_alpha_splits
        self._unit_nvme_para_alpha_splits = para._unit_nvme_alpha_splits

        self._backward_numel = \
            self._mb_gpu_para_alpha_splits[0] + \
            self._mb_cpu_para_alpha_splits[0] + \
            self._mb_nvme_para_alpha_splits[0]
        self._forward_numel = \
            self._mb_gpu_para_alpha_splits[1] + \
            self._mb_cpu_para_alpha_splits[1] + \
            self._mb_nvme_para_alpha_splits[1]

        self._backward_splits = _get_split_numels(
            self._backward_numel, opts_cpu_nvme_ratio, num_levels=2
        )
        self._forward_splits = _get_split_numels(
            self._forward_numel, opts_cpu_nvme_ratio, num_levels=2
        )
        # End of configuration.

        # Used for accumulating / transferring backward gradients.
        self._gpu_bwd_grads_buffer = RotateContainer(
            _allocate_memory_chunks(
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

        # TEMP
        self._data_stream = get_data_stream()
        self._grad_partition = _allocate_memory_chunks(
            self._unit_numel // dist.get_world_size(), self._num_units,
            gradacc_dtype, torch.cuda.current_device()
        )

    # TMEP
    def _calculate_global_grad_norm(self):
        # Calculate the global gradient norm.
        global_grad_norm = torch.tensor([0.], device=torch.cuda.current_device())
        for unit in reversed(range(self._num_units)):
            global_grad_norm += self._grad_partition[unit].norm() ** 2
            dist.print_rank0(f"Rank {dist.get_rank()} layer {unit} grad norm: {global_grad_norm.item()}")
        dist.all_reduce(global_grad_norm, op=dist.ReduceOp.SUM)
        dist.print_rank0(global_grad_norm.item())
        return global_grad_norm.item()

    # def _bind_ckpt_memory(self):
    #     # If already bound, return.
    #     if self._ckpt_bound:
    #         return
    #     self._ckpt_bound = True

    #     # Ensure that the interlayer coordinator is initialized.
    #     interlayer = get_interlayer_coordinator()
    #     assert interlayer.is_initialized, (
    #         "Interlayer coordinator must be initialized before init_optimizer."
    #     )

    #     # Used for storing forward gradients.
    #     self._gpu_fwd_grads_storage = interlayer.gpu_ckpt_base.view(
    #         dtype=get_flextrain_config().mixed_precision.gradacc_dtype
    #     )
    #     self._cpu_fwd_grads_storage = interlayer.cpu_ckpt_base.view(
    #         dtype=get_flextrain_config().mixed_precision.gradacc_dtype
    #     )

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

    def _submit_transfer_grads(self, unit_index: int):
        """ Launch the async IO operation to transfer gradients for the unit.

        Args:
            unit_index (int): Index of the unit.

        Returns:
            None
        """
        # Check if the unit is valid.
        if self._is_invalid_unit(unit_index):
            return

        # 1. Locate the target memory.
        # 2. Conduct all-reduce into tensor if necessary.
        # 3. Copy parameters from three resources:
        #    - GPU part to GPU optimizer working buffer
        #    - CPU part to CPU optimizer working buffer
        default_stream = torch.cuda.current_stream()

        def transfer_grads():
            # Locate the target memory.
            src_full_grads: torch.Tensor = self._gpu_bwd_transfer_grads

            # Synchonize with the default stream for the first unit.
            # Because there is no torch.cuda.synchronize() before.
            if unit_index == 0:
                default_stream.synchronize()

            # All-reduce the gradients.
            mem_partition = torch.chunk(
                src_full_grads, dist.get_world_size()
            )[dist.get_rank()]
            dist.reduce_scatter(
                mem_partition, src_full_grads,
                dist.ReduceOp.AVG
            )

            # Copy the gradients to the optimizer working buffer.
            self._grad_partition[unit_index].copy_(mem_partition)

        # Submit the task to the data stream.
        self._data_stream.submit(transfer_grads)

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

        # TODO: submit gradient transfer task.

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
        self._submit_transfer_grads(unit_index + 1)
        self._prepare_unit_grads(unit_index)

    def clear_backward_pipeline(self):
        """
        Cleanup the backward pipeline.
        """
        self._gpu_bwd_grads_buffer.rotate()
        self._submit_transfer_grads(0)
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
