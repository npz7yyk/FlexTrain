import torch

from math import ceil
from torch import Tensor
from torch.nn import Parameter
from typing import SupportsIndex, Iterable, Tuple, List, Dict

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
    FunctionHandle,
    FusedHandle
)
from flextrain.memory.nvme_swapper import AsyncNVMeSwapper
from flextrain.utils import dist
from flextrain.utils.logging import rank0_logger


class ItemPair:
    def __init__(self, items: Tuple):
        assert len(items) == 2
        self.items = list(items)

    def __getitem__(self, index: SupportsIndex):
        return self.items[index]

    def swap(self):
        self.items.reverse()


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


class FlexTrainModelCoordinator:

    def __init__(self):
        # Lazy initialization of model coordinator.
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
        self.num_units = 0

        # CUDA streams for async IO operations.
        self._data_stream = torch.cuda.Stream()

        # Lazy initialization of NVMe swapper.
        self._nvme_swapper = None

        # Map of unit index to its parameters.
        self._unit_parameters: Dict[int, ContiguousParaGroup] = {}

        # How to split the training data.
        each_rank_numel = contiguous_partitioned_numel(parameters)
        self._para_numels = self._get_split_numels(
            each_rank_numel, config.parameter_split_ratio
        )
        self._grad_numels = self._para_numels
        # Memory for optimizer states needed to be lazy allocated.

        # End of coordinator configurations.

        # Allocate parameter base containers.
        self._gpu_para_base: List[Tensor] = []
        self._cpu_para_base: List[Tensor] = []

        # Allocate GPU working memory for parameters.
        self._gpu_full_paras = ItemPair(_allocate_memory_chunks(
            contiguous_allgathered_numel(parameters), 2,
            self._device_dtype, torch.cuda.current_device()
        ))

        # Allocate NVMe prefetch buffer in CPU memory.
        self._nvme_prefetch_buffer = ItemPair(_allocate_memory_chunks(
            self._para_numels[2], 2,
            self._device_dtype, torch.device('cpu')
        ))

        # Handle for async IO operations.
        self._inflight_layer_handle: Tuple[int, Waitable] = (-1, None)

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

    def _is_invalid_unit(self, unit_index: int):
        return unit_index < 0 or unit_index >= self.num_units

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

        # Update the number of units.
        self.num_units += 1

        # Allocate parameter bases
        self._gpu_para_base.append(torch.empty(
            self._para_numels[0],
            dtype=self._device_dtype,
            device=torch.cuda.current_device()
        ))
        self._cpu_para_base.append(torch.empty(
            self._para_numels[1],
            dtype=self._device_dtype,
            device=torch.device('cpu')
        ))

        # Track the unit parameters.
        self._unit_parameters[unit_index] = ContiguousParaGroup(unit_paras)

        # Using GPU working window to conduct the broadcast.
        temp_gpu_buffer = self._gpu_inflight_paras

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

        # Detach the parameters from the memory.
        self._detach_unit_parameters(unit_index)

    def log_configuration(self):
        """
        Log the model coordinator configurations after initialization.
        """
        each_rank_numel = sum(self._para_numels)
        unit_numel = each_rank_numel * dist.get_world_size()

        # Log some useful information.
        rank0_logger.info(
            "\n\n> "
            f"FlexTrain model coordinator initialized with configurations:\n"
            f"  - device dtype: {self._device_dtype}\n"
            f"  - master dtype: {self._master_dtype}\n"
            f"  - number of units: {self.num_units}\n"
            f"  - unit parameter numel: {unit_numel}\n"
            f"  - each rank numel: {each_rank_numel}\n"
            f"  - parameter split numels: {self._para_numels}\n"
        )

    def _async_prepare_cpu_paras(self, unit_index: int):
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

        return self._nvme_reload(
            FlexTrainDataID(unit_index, Dtype.PARA),
            self._nvme_inflight_paras, async_op=True
        )

    def _async_prepare_gpu_paras(self, unit_index: int):
        """
        Prepare the GPU parameters for the unit.
        The GPU parameters are prepared in _gpu_inflight_paras.
        """
        # 0. Check if the unit is valid.
        if self._is_invalid_unit(unit_index):
            return DummyHandle()

        # 1. Locate the target memory.
        tar_full_paras = self._gpu_inflight_paras
        mem_partitions = torch.chunk(tar_full_paras, dist.get_world_size())
        mem_partition = mem_partitions[dist.get_rank()]

        # 2. Copy parameters from three resources:
        #    - GPU part from GPU base
        #    - CPU part from CPU base
        #    - NVMe part from CPU available buffer
        # 3. Conduct all-gather into tensor if necessary.
        gpu_view, cpu_view, nvme_view = torch.split(
            mem_partition, self._para_numels
        )
        with torch.cuda.stream(self._data_stream):
            gpu_view.copy_(self._gpu_para_base[unit_index], True)
            cpu_view.copy_(self._cpu_para_base[unit_index], True)
            nvme_view.copy_(self._nvme_available_paras, True)
            dist.all_gather(tar_full_paras, mem_partition, async_op=True)

        return FunctionHandle(torch.cuda.synchronize)

    def _submit_prepare_paras(self, unit_index: int):
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
        cpu_handle = self._async_prepare_cpu_paras(unit_index + 1)

        # Prepare the GPU parameters.
        gpu_handle = self._async_prepare_gpu_paras(unit_index)

        # Keep track of the inflight operation.
        self._inflight_layer_handle = (
            unit_index, FusedHandle([cpu_handle, gpu_handle])
        )

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
        inflight_unit, handle = self._inflight_layer_handle
        assert inflight_unit == unit_index, (
            f"Async IO operation is not for this unit: "
            f"unit_index={unit_index} != inflight_unit={inflight_unit}"
        )
        handle.wait()

        # Just available buffer is now the prefetch buffer.
        # Just inflight buffer is now available after handle.wait().
        self._nvme_prefetch_buffer.swap()

        # Just inflight GPU buffer is now available for unit forward.
        # Just available GPU buffer is now free for prefetching.
        self._gpu_full_paras.swap()

        # Link the parameters to the available buffer.
        self._link_unit_parameters(unit_index, self._gpu_available_paras)

    def warmup_forward_pipeline(self):
        """
        Warm up the forward pipeline.
        Recommendation: call before LLM pre_processing for further speedup.
        """

        # Load the first unit parameters to CPU.
        self._async_prepare_cpu_paras(0).wait()
        self._nvme_prefetch_buffer.swap()

        # Launch the first unit forward.
        self._submit_prepare_paras(0)

    def pre_forward_unit(self, unit_index: int):
        """
        Prepare the unit for forward pass.

        Functions:
        1. Ensure the availability of the parameters and checkpoint.
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

    def pre_backward_unit(self, unit_index: int):
        """
        Prepare the unit for backward pass.

        Functions:
        1. Ensure the availability of the parameters and checkpoint.
        2. Kick off relevant prefetching tasks if needed.

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
        self._submit_prepare_paras(unit_index - 1)


_MODEL_COORDINATOR = FlexTrainModelCoordinator()


def get_model_coordinator():
    """
    Get the model coordinator.

    Returns:
        FlexTrainModelCoordinator: The model coordinator.
    """
    return _MODEL_COORDINATOR


class FlexTrainOptCoordinator:

    def __init__(self):
        # Lazy initialization of optimizer coordinator.
        self._initialized = False

    def _init_coordinator(self, optimizer: torch.optim.Optimizer):
        # If already initialized, return.
        if self._initialized:
            return
        self._initialized = True

        ...


_OPT_COORDINATOR = FlexTrainOptCoordinator()


def get_opt_coordinator():
    """
    Get the optimizer coordinator.

    Returns:
        FlexTrainOptCoordinator: The optimizer coordinator.
    """
    return _OPT_COORDINATOR
