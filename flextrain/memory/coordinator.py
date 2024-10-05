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

    def pre_forward_micro_batch(self, unit_index: int, micro_batch_index: int):
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

    def pre_backward_micro_batch(
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

    def pre_forward_unit(self, unit_index: int):
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

    def pre_backward_unit(self, unit_index: int):
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
            f"  - checkpoint splits: {self._ckpt_numels}\n"
            f"  - gradient splits: {self._grad_numels}\n"
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

    def _sync_pre_forward_micro_batch(self):
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

    def _submit_pre_forward_micro_batch(
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

    def pre_forward_micro_batch(
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

        self._sync_pre_forward_micro_batch()
        self._submit_pre_forward_micro_batch(ckpt_prefetch, ckpt_offload)

    def _sync_pre_backward_micro_batch(self):
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

    def _submit_pre_backward_micro_batch(
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

    def pre_backward_micro_batch(
        self,
        ckpt_prefetch: InterLayerTask = None,
        grad_prefetch: InterLayerTask = None,
        grad_offload: InterLayerTask = None
    ):
        # Mask invalid tasks.
        ckpt_prefetch = self._mask_invalid_task(ckpt_prefetch)
        grad_prefetch = self._mask_invalid_task(grad_prefetch)
        grad_offload = self._mask_invalid_task(grad_offload)

        self._sync_pre_backward_micro_batch()
        self._submit_pre_backward_micro_batch(
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

    def _copy_master_parameters(self):
        assert False, "Not implemented yet."
        # Unpack numels
        gpu_opt_numel, cpu_opt_numel, nvme_opt_numel = self._opt_numels
        gpu_mdl_numel, cpu_mdl_numel, nvme_mdl_numel = self._model_numels
        assert gpu_opt_numel <= gpu_mdl_numel

        # Create memory for master parameters and optimizer states.
        # Set to zero for fast optimizer initialization.
        self._gpu_master_opts = _allocate_memory_chunks(
            gpu_opt_numel * self._each_numel_num_states, self._num_units,
            self._master_dtype, torch.cuda.current_device()
        )
        self._cpu_master_opts = _allocate_memory_chunks(
            cpu_opt_numel * self._each_numel_num_states, self._num_units,
            self._master_dtype, torch.device('cpu')
        )

        def _get_para_in_opt(_tensor: Tensor) -> Tensor:
            # Get the master parameters in the contiguous optimizer states.
            return torch.chunk(_tensor, self._each_numel_num_states)[0]

        # temp buffer for CPU + NVMe device dtype parameters.
        # Note that it is each_numel_num_states + 1 times larger.
        temp_cpu_buffer: Tensor = self._cpu_work_opt_states

        para_coordinator = get_para_coordinator()

        # Copy the master parameters from the device dtype parameters.
        for i, unit in enumerate(self._train_units):
            # 1. Copy GPU master parameters.
            # Note: GPU master parameters ratio <= GPU model parameters ratio,
            #       i.e. gpu_opt_numel <= gpu_mdl_numel.
            _get_para_in_opt(self._gpu_master_opts[i]).copy_(
                para_coordinator._gpu_para_base[unit][:gpu_opt_numel]
            )

            # 2. Copy CPU + NVMe device dtype parameters.
            part1 = gpu_mdl_numel - gpu_opt_numel
            part2 = part1 + cpu_mdl_numel
            part3 = part2 + nvme_mdl_numel
            assert part3 == cpu_opt_numel + nvme_opt_numel
            # Set to zero for fast optimizer initialization.
            temp_cpu_buffer.zero_()
            temp_cpu_buffer[:part1].copy_(
                para_coordinator._gpu_para_base[unit][gpu_opt_numel:]
            )
            temp_cpu_buffer[part1:part2].copy_(
                para_coordinator._cpu_para_base[unit]
            )
            # We need to use device_dtype buffer for reloading.
            _nvme_reload(
                FlexTrainDataID(unit, Dtype.PARA),
                para_coordinator._nvme_available_paras
            )
            temp_cpu_buffer[part2:part3].copy_(
                para_coordinator._nvme_available_paras
            )

            # 3. Store CPU + NVMe master parameters.
            _get_para_in_opt(self._cpu_master_opts[i]).copy_(
                temp_cpu_buffer[:cpu_opt_numel]
            )
            # Note that NVMe OPTS is each_numel_num_states + 1 times larger.
            start = cpu_opt_numel
            end = start + nvme_opt_numel * self._each_numel_num_states
            _nvme_offload(
                FlexTrainDataID(unit, Dtype.OPTS),
                temp_cpu_buffer[start:end]
            )

    def log_configuration(self):
        rank0_logger.info(
            "\n\n> "
            f"FlexTrain optimizer coordinator initialized with configurations:"
            f"\n"
            f"  - device dtype: {self._device_dtype}\n"
            f"  - master dtype: {self._master_dtype}\n"
            f"  - number of units under training: {self._num_units}\n"
            f"  - optimizer split numels: {self._opt_numels}\n"
        )

    def _is_invalid_unit(self, unit_index: int):
        return unit_index < 0 or unit_index >= self._num_units

    def initialize(
        self,
        train_units: List[int],
        each_numel_num_states: int = 2
    ):
        """
        Initialize FlexTrain optimizer from assigned parameter groups.
        Must be called after the parameter coordinator is initialized.

        Args:
            train_units (List[int]): List of unit indices under training.
            each_numel_num_states (int, optional): \
                Typical optimization is conducted element-wise. \
                This argument specifies the number of optimizer states \
                for each parameter element. If not provided, default to 2, \
                which is the most common case (e.g. Adam, AdamW). (default: 2)

        Returns:
            None
        """

        # 0. Before initialization:
        # Check if the parameter coordinator is initialized.
        para_coordinator = get_para_coordinator()
        assert para_coordinator.is_initialized, (
            "Parameter coordinator must be initialized before init_optimizer."
        )
        # Check if the optimizer coordinator is not initialized.
        assert not self._initialized, (
            "Optimizer coordinator is already initialized."
        )
        # Link to units under training.
        self._train_units = sorted(train_units)
        self._num_units = len(self._train_units)
        self._unit_parameters = para_coordinator.unit_parameter_map

        # CUDA streams for async IO operations.
        self._data_stream = torch.cuda.Stream()

        # 1. Set the configuration for the optimizer.
        config = get_flextrain_config()
        self._device_dtype = config.mixed_precision.device_dtype
        self._master_dtype = config.mixed_precision.master_dtype

        # numel_per_rank = para_coordinator.numel_per_rank
        self._unit_numel = para_coordinator.aligned_unit_numel
        # self._model_numels = para_coordinator.model_split_numels
        # self._opt_numels = _get_split_numels(
        #     numel_per_rank, config.split_ratio.optimizer
        # )
        # self._grad_numels = (
        #     self._opt_numels[0],  # GPU optimizer
        #     self._opt_numels[1] + self._opt_numels[2]  # CPU optimizer
        # )
        # assert self._model_numels[0] >= self._opt_numels[0], \
        #     "GPU parameter ratio should be larger than GPU optimizer ratio."
        # # Plus one for the master parameters.
        # self._each_numel_num_states = each_numel_num_states + 1

        # Split the optimizer states.
        # We have algorithm splits and memory splits:
        # - algorithm splits: how different states are grouped.
        # - memory splits: how the data is stored in memory hierarchy.
        # gpu_numel = self._opt_numels[0]  # GPU optimizer
        # cpu_numel = self._opt_numels[1] + self._opt_numels[2]  # CPU optimizer
        # self._gpu_alg_splits = [gpu_numel] * self._each_numel_num_states
        # self._cpu_alg_splits = [cpu_numel] * self._each_numel_num_states
        # self._cpu_mem_splits = [
        #     self._opt_numels[1] * self._each_numel_num_states,
        #     self._opt_numels[2] * self._each_numel_num_states
        # ]

        # 2. Log the optimizer coordinator configurations.
        self.log_configuration()

        # 3. Allocate working memory.
        # a. We need 3 buffers for CPU optimizer states:
        #    prefetch buffer, working buffer, commit buffer.
        # self._cpu_opts_buffer = RotateContainer(_allocate_memory_chunks(
        #     cpu_numel * each_numel_num_states, 3,
        #     self._master_dtype, torch.device('cpu')
        # ))
        # b. GPU gradients buffer for backwarding, working at device precision.
        #    receiving buffer, transferring buffer.
        self._gpu_bwd_grads_buffer = RotateContainer(_allocate_memory_chunks(
            self._unit_numel, 2,
            self._device_dtype, torch.cuda.current_device()
        ))
        # c. GPU gradients buffer for optimizer, working at master precision.
        #    receiving buffer, working buffer.
        # self._gpu_opt_grads_buffer = RotateContainer(_allocate_memory_chunks(
        #     self._grad_numels[0], 2,
        #     self._master_dtype, torch.cuda.current_device()
        # ))

        # 4. Copy the master parameters.
        # self._copy_master_parameters()

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

        # Get the gradients.
        grad_buffer = self._gpu_bwd_receive_grads
        torch.zero_(grad_buffer)

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

        # Gradients are ready for transfer.
        self._gpu_bwd_grads_buffer.rotate()

        # 1. Locate the target memory.
        # 2. Conduct all-reduce into tensor if necessary.
        # 3. Copy parameters from three resources:
        #    - GPU part to GPU optimizer working buffer
        #    - CPU part to CPU optimizer working buffer
        default_stream = torch.cuda.current_stream()
        with torch.cuda.stream(self._data_stream):
            # Locate the target memory.
            src_full_grads: torch.Tensor = self._gpu_bwd_transfer_grads
            mem_partitions = torch.chunk(src_full_grads, dist.get_world_size())
            mem_partition = mem_partitions[dist.get_rank()]
            # gpu_view, cpu_view = torch.split(
            #     mem_partition, self._grad_numels
            # )

            # default_stream.synchronize()

            dist.reduce_scatter(
                mem_partition, src_full_grads,
                dist.ReduceOp.AVG, async_op=True
            )

        self._inflight_grad_handle = (
            unit_index, DummyHandle()
        )

    def _synchronize_transfer_grads(self, unit_index: int):
        """ Synchronize the transfer of gradients for the unit.

        Args:
            unit_index (int): Index of the unit.

        Returns:
            None
        """
        # Check if the unit is valid.
        if self._is_invalid_unit(unit_index):
            return

        # Wait for the async IO operation to finish.
        inflight_unit, handle = self._inflight_grad_handle
        assert inflight_unit == unit_index, (
            f"Async IO operation is not for this unit: "
            f"unit_index={unit_index} != inflight_unit={inflight_unit}"
        )
        handle.wait()

    def pre_backward_unit(self, unit_index: int):
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

        self._synchronize_transfer_grads(unit_index + 2)
        self._submit_transfer_grads(unit_index + 1)
        self._prepare_unit_grads(unit_index)

    def clear_backward_pipeline(self):
        """
        Cleanup the backward pipeline.
        """
        self._synchronize_transfer_grads(1)
        self._submit_transfer_grads(0)
        torch.cuda.synchronize()


_OPT_COORDINATOR = FlexTrainOptCoordinator()


def get_opt_coordinator():
    """
    Get the optimizer coordinator.

    Returns:
        FlexTrainOptCoordinator: The optimizer coordinator.
    """
    return _OPT_COORDINATOR
