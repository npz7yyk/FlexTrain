import torch

from math import ceil
from torch import Tensor
from torch.nn import Parameter
from typing import SupportsIndex, Iterable, Dict

from flextrain.config import get_flextrain_config
from flextrain.memory import (
    FlexTrainDataType as Dtype,
    FlexTrainDataID,
    contiguous_allgathered_numel,
    contiguous_partitioned_numel,
    move_into_contiguous,
    ContiguousParaGroup,
    Waitable,
    DummyHandle,
    AsyncIOHandle,
    FusedHandle
)
from flextrain.memory.nvme_swapper import AsyncNVMeSwapper
from flextrain.utils import dist
from flextrain.utils.logging import rank0_logger


class LoopIndexer:
    """
    LoopIndexer is a class that wraps a target that supports index.
    It provides a way to loop through the target in a circular manner.
    """

    def __init__(self, target: SupportsIndex):
        self._target = target
        self._total_items = len(target)

        self._index_offset = 0

    def step(self):
        self._index_offset += 1
        if self._index_offset >= self._total_items:
            self._index_offset = 0

    def __getitem__(self, index: int):
        curr_index = (index + self._index_offset) % self._total_items
        return self._target[curr_index]


def _allocate_memory_chunks(
    numel: int,
    chunks: int,
    dtype: torch.dtype,
    device: torch.device
):
    device = torch.device(device)
    mem = torch.empty(
        numel * chunks,
        dtype=dtype,
        device=device,
        pin_memory=True if device.type == 'cpu' else False
    )
    return torch.chunk(mem, chunks)


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


class FlexTrainMemoryCoordinator:

    def __init__(self):
        # Lazy initialization of memory coordinator.
        self._initialized = False

    def _get_split_numels(self, total_numel: int, ratios: Iterable[float]):
        # Ensure the number of levels is 2.
        NUM_LEVELS = 3
        if len(ratios) == NUM_LEVELS:
            ratios = ratios[:NUM_LEVELS - 1]

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

        # Whether running in single GPU mode.
        self._single_gpu = dist.get_world_size() == 1

        # Mixed precision dtype for accelerator.
        self._device_dtype = config.device_dtype

        # Mixed precision dtype for master.
        self._master_dtype = config.master_dtype

        # Number of units in the model.
        layers = config.num_layers
        interval = config.checkpoint_interval
        self._num_units = (layers + interval - 1) // interval

        # Lazy initialization of NVMe swapper.
        self._nvme_swapper = None

        # Map of unit index to its parameters.
        self._unit_parameters: Dict[int, ContiguousParaGroup] = {}

        # How to split the training data.
        each_rank_numel = contiguous_partitioned_numel(parameters)
        # TODO: cut activation checkpoints
        self._para_numels = self._get_split_numels(
            each_rank_numel, config.parameter_split_ratio
        )
        self._grad_numels = self._para_numels
        # Memory for optimizer states needed to be lazy allocated.
        # TODO: cut optimizer states

        # Log some useful information.
        unit_numel = sum(p.numel() for p in parameters)
        rank0_logger.info(
            "\n\n> "
            f"FlexTrain Memory coordinator initialized with configurations:\n"
            f"  - device dtype: {self._device_dtype}\n"
            f"  - master dtype: {self._master_dtype}\n"
            f"  - number of units: {self._num_units}\n"
            f"  - unit parameter numel: {unit_numel}\n"
            f"  - each rank numel: {each_rank_numel}\n"
            f"  - parameter split numels: {self._para_numels}\n"
        )

        # End of coordinator configurations.

        # Allocate GPU working window.
        self._gpu_working_windows = LoopIndexer(
            _allocate_memory_chunks(
                contiguous_allgathered_numel(parameters),
                2, self._device_dtype, torch.cuda.current_device()
            )
        )

        # Allocate parameter bases
        self._gpu_para_base = _allocate_memory_chunks(
            self._para_numels[0], self._num_units,
            self._device_dtype, torch.cuda.current_device()
        )
        self._cpu_para_base = _allocate_memory_chunks(
            self._para_numels[1], self._num_units,
            self._device_dtype, torch.device('cpu')
        )

        # Handle for async IO operations.
        self._last_layer_index_handle = (-1, None)

    def _nvme_offload(
        self,
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
        if self._nvme_swapper is None:
            self._nvme_swapper = AsyncNVMeSwapper(
                get_flextrain_config().nvme_swap_dir
            )

        # Call the NVMe swapper.
        return self._nvme_swapper.swap_out(
            data_ids, tensors, async_op
        )

    def _nvme_reload(
        self,
        data_ids: FlexTrainDataID | Iterable[FlexTrainDataID],
        tensors: Tensor | Iterable[Tensor],
        async_op: bool = False
    ):
        # If swapper is not initialized, return.
        if self._nvme_swapper is None:
            return DummyHandle() if async_op else None

        # Filter out empty tensors.
        data_ids, tensors = _filter_tensors(data_ids, tensors)

        # If no tensors to reload, return.
        if len(data_ids) == 0:
            return DummyHandle() if async_op else None

        # Call the NVMe swapper.
        return self._nvme_swapper.swap_in(
            data_ids, tensors, async_op
        )

    def init_unit_parameters(
        self,
        unit_index: int,
        unit_paras: Iterable[Parameter]
    ):
        """
        Initialize the memory for the parameters in a unit.
        Parameters are moved into contiguous memory and partitioned.
        The contiguous memory is partitioned as:
        [(GPU, CPU, NVMe), (GPU, CPU, NVMe), ... (GPU, CPU, NVMe)].
        Each (GPU, CPU, NVMe) is owned by a rank.

        Args:
            unit_index (int): Index of the unit.
            unit_parameters (List[Parameter]): \
                List of parameters in a unit.

        Returns:
            None
        """

        # Initialize memory allocation if not done.
        self._init_coordinator(unit_paras)

        # Track the unit parameters.
        self._unit_parameters[unit_index] = ContiguousParaGroup(unit_paras)

        # Using GPU working window to conduct the broadcast.
        temp_gpu_buffer = self._gpu_working_windows[0]

        # Move parameters into contiguous memory.
        move_into_contiguous(unit_paras, temp_gpu_buffer)

        # Broadcast the parameters.
        dist.broadcast(temp_gpu_buffer, src=0)

        # Partition parameters.
        partitioned_paras = torch.chunk(
            temp_gpu_buffer, dist.get_world_size()
        )[dist.get_rank()]

        # Get GPU, CPU and NVMe views of partitioned parameters.
        gpu_view, cpu_view, nvme_view = torch.split(
            partitioned_paras, self._para_numels
        )

        # Store the views.
        self._gpu_para_base[unit_index].copy_(gpu_view)
        self._cpu_para_base[unit_index].copy_(cpu_view)
        self._nvme_offload(
            FlexTrainDataID(unit_index, Dtype.PARA),
            nvme_view
        )

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

        # Get the unit parameters.
        unit_paras = self._unit_parameters[unit_index]

        # Link the parameters.
        unit_paras.link_para_to(buffer)

    def pre_forward_layer(self, unit_index: int):
        """
        Prepare the layer for forward pass.

        Functions:
        1. Ensure the availability of the parameters and checkpoint.
        2. Kick off relevant prefetching tasks.

        Args:
            unit_index (int): Index of the unit.
        """
        

    def post_forward_layer(self, unit_index: int):
        pass

    def _single_gpu_forward_prepare_layer(self, unit_index: int):
        """
        Prepare the layer for forward pass in single GPU mode.
        The parameters are linked to the buffer.

        Args:
            unit_index (int): Index of the unit.

        Returns:
            None
        """
        # Get the buffer.
        buffer = self._gpu_working_windows[0]

        # Link the parameters.
        self._link_unit_parameters(unit_index, buffer)

    def _multi_gpu_forward_prepare_layer(self, unit_index: int):
        """
        Prepare the layer for forward pass in multi GPU mode.
        The parameters are linked to the buffer.

        Args:
            unit_index (int): Index of the unit.

        Returns:
            None
        """
        pass


_MEMORY_COORDINATOR = FlexTrainMemoryCoordinator()


def get_memory_coordinator():
    """
    Get the memory coordinator.

    Returns:
        FlexTrainMemoryCoordinator: The memory coordinator.

    Raises:
        AssertionError: If the memory coordinator is not initialized.
    """
    assert _MEMORY_COORDINATOR is not None, \
        "Memory coordinator not initialized"
    return _MEMORY_COORDINATOR
