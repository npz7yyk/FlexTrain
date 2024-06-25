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


class RotateContainer:
    def __init__(self, items: Tuple):
        self._items = list(items)

    def __getitem__(self, index: SupportsIndex):
        return self._items[index]

    def rotate(self):
        self._items.append(self._items.pop(0))


def _allocate_memory_chunks(
    numel: int,
    chunks: int,
    dtype: torch.dtype,
    device: torch.device,
    set_zero: bool = False
):
    device = torch.device(device)
    mem = torch.empty(
        numel * chunks,
        dtype=dtype,
        device=device,
        pin_memory=True if device.type == 'cpu' else False
    )
    if set_zero:
        mem.zero_()
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


def _get_split_numels(total_numel: int, ratios: Iterable[float]):
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
        _NVME_SWAPPER = AsyncNVMeSwapper(get_flextrain_config().nvme_swap_dir)

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


class FlexTrainModelCoordinator:

    def __init__(self):
        # Lazy initialization of model coordinator.
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

        # Whether running in single GPU mode.
        self._single_gpu = dist.get_world_size() == 1

        # Mixed precision dtype for accelerator.
        self._device_dtype = config.device_dtype

        # Mixed precision dtype for master.
        self._master_dtype = config.master_dtype

        # Number of units in the model.
        self._num_units = 0

        # CUDA streams for async IO operations.
        self._data_stream = torch.cuda.Stream()

        # Lazy initialization of NVMe swapper.
        self._nvme_swapper = None

        # Map of unit index to its parameters.
        self._unit_parameters: Dict[int, ContiguousParaGroup] = {}

        # How to split the training data.
        numel_per_rank = contiguous_partitioned_numel(parameters)
        self._para_numels = _get_split_numels(
            numel_per_rank, config.parameter_split_ratio
        )
        self._grad_numels = self._para_numels
        self._numel_per_rank = numel_per_rank
        self._original_unit_numel = sum(p.numel() for p in parameters)
        # Memory for optimizer states needed to be lazy allocated.

        # End of coordinator configurations.

        # Allocate parameter base containers.
        self._gpu_para_base: List[Tensor] = []
        self._cpu_para_base: List[Tensor] = []

        # Allocate GPU working memory for parameters.
        self._gpu_full_paras = RotateContainer(_allocate_memory_chunks(
            contiguous_allgathered_numel(parameters), 2,
            self._device_dtype, torch.cuda.current_device()
        ))

        # Allocate NVMe prefetch buffer in CPU memory.
        self._nvme_prefetch_buffer = RotateContainer(_allocate_memory_chunks(
            self._para_numels[2], 2,
            self._device_dtype, torch.device('cpu')
        ))

        # Handle for async IO operations.
        self._inflight_layer_handle: Tuple[int, Waitable] = (-1, None)

    @property
    def is_initialized(self):
        return self._initialized

    @property
    def num_units(self):
        return self._num_units

    @property
    def unit_parameter_map(self):
        return self._unit_parameters

    @property
    def numel_per_rank(self):
        return self._numel_per_rank

    @property
    def model_split_numels(self):
        return self._para_numels

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
        self._num_units += 1

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
        _nvme_offload(
            FlexTrainDataID(unit_index, Dtype.PARA),
            nvme_view
        )

        # Detach the parameters from the memory.
        self._detach_unit_parameters(unit_index)

    def log_configuration(self):
        """
        Log the model coordinator configurations after initialization.
        """
        rank0_logger.info(
            "\n\n> "
            f"FlexTrain model coordinator initialized with configurations:\n"
            f"  - device dtype: {self._device_dtype}\n"
            f"  - master dtype: {self._master_dtype}\n"
            f"  - number of units: {self._num_units}\n"
            f"  - unit parameter numel: {self._original_unit_numel}\n"
            f"  - each rank numel: {self._numel_per_rank}\n"
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

        return _nvme_reload(
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
        self._async_prepare_cpu_paras(0).wait()
        self._nvme_prefetch_buffer.rotate()

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

    @property
    def is_initialized(self):
        return self._initialized

    @property
    def _cpu_prefetch_opt_states(self):
        return self._cpu_opts_buffer[0]

    @property
    def _cpu_work_opt_states(self):
        return self._cpu_opts_buffer[1]

    @property
    def _cpu_commit_opt_states(self):
        return self._cpu_opts_buffer[2]

    @property
    def each_numel_num_states(self):
        return self._each_numel_num_states - 1

    def get_gpu_unit_states(self, unit_index: int):
        mem_index = self._train_units.index(unit_index)
        return torch.chunk(
            self._gpu_master_opts[mem_index],
            self._each_numel_num_states
        )

    def _copy_master_parameters(self):
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

        model_coordinator = get_model_coordinator()

        # Copy the master parameters from the device dtype parameters.
        for i, unit in enumerate(self._train_units):
            # 1. Copy GPU master parameters.
            # Note: GPU master parameters ratio <= GPU model parameters ratio,
            #       i.e. gpu_opt_numel <= gpu_mdl_numel.
            _get_para_in_opt(self._gpu_master_opts[i]).copy_(
                model_coordinator._gpu_para_base[unit][:gpu_opt_numel]
            )

            # 2. Copy CPU + NVMe device dtype parameters.
            part1 = gpu_mdl_numel - gpu_opt_numel
            part2 = part1 + cpu_mdl_numel
            part3 = part2 + nvme_mdl_numel
            assert part3 == cpu_opt_numel + nvme_opt_numel
            # Set to zero for fast optimizer initialization.
            temp_cpu_buffer.zero_()
            temp_cpu_buffer[:part1].copy_(
                model_coordinator._gpu_para_base[unit][gpu_opt_numel:]
            )
            temp_cpu_buffer[part1:part2].copy_(
                model_coordinator._cpu_para_base[unit]
            )
            # We need to use device_dtype buffer for reloading.
            _nvme_reload(
                FlexTrainDataID(unit, Dtype.PARA),
                model_coordinator._nvme_available_paras
            )
            temp_cpu_buffer[part2:part3].copy_(
                model_coordinator._nvme_available_paras
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

    def initialize(
        self,
        train_units: List[int],
        each_numel_num_states: int = 2
    ):
        """
        Initialize FlexTrain optimizer from assigned parameter groups.
        Must be called after the model coordinator is initialized.

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
        # Check if the model coordinator is initialized.
        assert get_model_coordinator().is_initialized, (
            "Model coordinator must be initialized before init_optimizer."
        )
        # Check if the optimizer coordinator is not initialized.
        assert not self._initialized, (
            "Optimizer coordinator is already initialized."
        )
        # Link to units under training.
        self._train_units = sorted(train_units)

        # 1. Set the configuration for the optimizer.
        self._device_dtype = get_flextrain_config().device_dtype
        self._master_dtype = get_flextrain_config().master_dtype

        self._num_units = len(self._train_units)

        numel_per_rank = get_model_coordinator().numel_per_rank
        self._model_numels = get_model_coordinator().model_split_numels
        self._opt_numels = _get_split_numels(
            numel_per_rank, get_flextrain_config().optimizer_split_ratio
        )
        assert self._model_numels[0] >= self._opt_numels[0], \
            "GPU parameter ratio should be larger than GPU optimizer ratio."
        # Plus one for the master parameters.
        self._each_numel_num_states = each_numel_num_states + 1

        # Split the optimizer states.
        # We have algorithm splits and memory splits:
        # - algorithm splits: how different states are grouped.
        # - memory splits: how the data is stored in memory hierarchy.
        gpu_numel = self._opt_numels[0]  # GPU optimizer
        cpu_numel = self._opt_numels[1] + self._opt_numels[2]  # CPU optimizer
        self._gpu_alg_splits = [gpu_numel] * self._each_numel_num_states
        self._cpu_alg_splits = [cpu_numel] * self._each_numel_num_states
        self._cpu_mem_splits = [
            self._opt_numels[1] * self._each_numel_num_states,
            self._opt_numels[2] * self._each_numel_num_states
        ]

        # 2. Log the optimizer coordinator configurations.
        self.log_configuration()

        # 3. Allocate prefetch buffer for optimizer states.
        # We need 3 buffers for CPU optimizer states:
        # prefetch buffer, work buffer, commit buffer.
        self._cpu_opts_buffer = RotateContainer(_allocate_memory_chunks(
            cpu_numel * each_numel_num_states, 3,
            self._master_dtype, torch.device('cpu')
        ))

        # 4. Copy the master parameters.
        self._copy_master_parameters()


_OPT_COORDINATOR = FlexTrainOptCoordinator()


def get_opt_coordinator():
    """
    Get the optimizer coordinator.

    Returns:
        FlexTrainOptCoordinator: The optimizer coordinator.
    """
    return _OPT_COORDINATOR
