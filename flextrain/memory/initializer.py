import functools
import os
import torch

from dataclasses import dataclass
from torch import Tensor
from torch.nn import Parameter, Module
from typing import Callable, Iterable, SupportsIndex, List

from flextrain.config import get_flextrain_config
# from flextrain.memory_coordinator import (
#     is_memory_coordinator_initialized,
#     init_memory_coordinator,
#     get_memory_coordinator,
#     FlexTrainMemoryCoordinator
# )
from flextrain.utils import dist
from flextrain.utils.logging import print_rank0
from flextrain.utils.memory import (
    move_into_contiguous
)


_IS_PARA_RANK_UNIQUE_FUNC = None


def set_is_para_rank_unique_func(func):
    global _IS_PARA_RANK_UNIQUE_FUNC
    assert callable(func), "Must provide a function for is_para_rank_unique"
    _IS_PARA_RANK_UNIQUE_FUNC = func


def is_para_rank_unique(x: Parameter):
    if dist.get_world_size() == 1:
        return False
    elif _IS_PARA_RANK_UNIQUE_FUNC is None:
        return False
    else:
        return _IS_PARA_RANK_UNIQUE_FUNC(x)


def group_paras_to_shared_and_unique(paras: Iterable[Parameter]):
    shared_paras: Iterable[Parameter] = []
    unique_paras: Iterable[Parameter] = []
    for para in paras:
        if is_para_rank_unique(para):
            unique_paras.append(para)
        else:
            shared_paras.append(para)
    return shared_paras, unique_paras


def contiguous_allgathered_numel(paras: Iterable[Parameter]):
    """ Allgathered numel of the contiguous memory for the parameters. """

    world_size = dist.get_world_size()

    shared_paras, unique_paras = group_paras_to_shared_and_unique(paras)
    total_shared_numel = sum(para.numel() for para in shared_paras)
    total_unique_numel = sum(para.numel() for para in unique_paras)

    # Align the total shared numel to the world size.
    if total_shared_numel % world_size:
        total_shared_numel += world_size - total_shared_numel % world_size

    return total_shared_numel + total_unique_numel


def contiguous_partitioned_numel(paras: Iterable[Parameter]):
    """ Partitioned numel of the contiguous memory for the parameters. """

    world_size = dist.get_world_size()

    shared_paras, unique_paras = group_paras_to_shared_and_unique(paras)
    total_shared_numel = sum(para.numel() for para in shared_paras)
    total_unique_numel = sum(para.numel() for para in unique_paras)

    # Align the total shared numel to the world size.
    if total_shared_numel % world_size:
        total_shared_numel += world_size - total_shared_numel % world_size

    return total_shared_numel // world_size + total_unique_numel


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


def _allocate_loop_indexed_memory(numel, chunks, dtype, device):
    device = torch.device(device)
    mem = torch.empty(
        numel * chunks,
        dtype=dtype,
        device=device,
        pin_memory=True if device.type == 'cpu' else False
    )
    return LoopIndexer(torch.chunk(mem, chunks))


_ALLOCATED_MEMORY = None


@dataclass
class FlexTrainMemory:
    """
    FlexTrain allocated memory on GPU, CPU and NVMe.
    """

    total_units: int
    unit_numel: int

    gpu_work_windows: LoopIndexer

    cpu: Tensor

    nvme_dir: str


def is_flextrain_memory_allocated():
    return _ALLOCATED_MEMORY is not None


def allocate_flextrain_memory(unit_numel: int):
    """
    Public API to allocate memory for FlexTrain. FlexTrain dist and config
    must be initialized before calling this function.
    GPU, CPU and NVMe memory will be allocated.

    Args:
        unit_numel (int):
            Total numel of a LLM model unit (may contain multiple layers).

    Returns:
        FlexTrainMem:
            The allocated memory.
    """

    global _ALLOCATED_MEMORY

    assert not is_flextrain_memory_allocated(), \
        "FlexTrain memory already allocated"

    config = get_flextrain_config()
    world_size = dist.get_world_size()

    # 1. Allocate GPU memory
    torch.cuda.empty_cache()
    gpu_work_windows = _allocate_loop_indexed_memory(
        unit_numel, 1, config.dtype, torch.cuda.current_device()
    )

    _ALLOCATED_MEMORY = FlexTrainMemory()


def get_flextrain_memory():
    """
    Public API to get the allocated memory for FlexTrain.
    flextrain.dist and config must be initialized before calling this function.

    Returns:
        FlexTrainMem:
            The allocated memory.
    """
    assert is_flextrain_memory_allocated(), "FlexTrain memory not allocated"
    return _ALLOCATED_MEMORY


# class ContiguousTensorGroup:
#     """
#     Manage a group of tensors in contiguous memory.

#     Args:
#         tensors (Iterable[Tensor]):
#             The tensors to manage.
#         target_contiguous_mem (Tensor, default=None):
#             The target contiguous memory to move the tensors into. \
#                 If not provided, tensors will not be moved.
#     """
#     def __init__(
#         self,
#         tensors: Iterable[Tensor],
#         target_contiguous_mem: Tensor = None
#     ):
#         self.tensors = list(tensors)
#         self.numels = [tensor.numel() for tensor in self.tensors]
#         self.shapes = [tensor.shape for tensor in self.tensors]

#         if target_contiguous_mem is not None:
#             move_into_contiguous(self.tensors, target_contiguous_mem)

#     def recover_view_on(self, contiguous_mem: Tensor):
#         offset = 0
#         for t, n, s in zip(self.tensors, self.numels, self.shapes):
#             t.data = contiguous_mem[offset: offset + n].view(s)
#             offset += n

#     def detach_view(self):
#         for tensor in self.tensors:
#             free_tensor(tensor)


# class ContiguousParaGroup:
#     """
#     Manage a group of parameters in contiguous memory.

#     Args:
#         tensors (Iterable[Parameter]):
#             The parameters to manage.
#         target_contiguous_mem (Tensor, default=None):
#             The target contiguous memory to move the tensors into. \
#                 If not provided, tensors will not be moved.
#     """
#     def __init__(
#         self,
#         paras: Iterable[Parameter],
#         target_contiguous_mem: Tensor = None
#     ):
#         # Group the parameters into unique and shared.
#         self.unique_paras, self.shared_paras = [], []
#         for para in paras:
#             if is_para_rank_unique(para):
#                 self.unique_paras.append(para)
#             else:
#                 self.shared_paras.append(para)
#         self.numels = [tensor.numel() for tensor in self.tensors]
#         self.shapes = [tensor.shape for tensor in self.tensors]

#         if target_contiguous_mem is not None:
#             move_into_contiguous(self.tensors, target_contiguous_mem)

#     def recover_view_on(self, contiguous_mem: Tensor):
#         offset = 0
#         for t, n, s in zip(self.tensors, self.numels, self.shapes):
#             t.data = contiguous_mem[offset: offset + n].view(s)
#             offset += n

#     def detach_view(self):
#         for tensor in self.tensors:
#             free_tensor(tensor)


_SHUTDOWN_INIT = False


def shutdown_init_context():
    """
    Shutdown the model initializer,
    i.e. recover the original layer init temporarily.
    Has no effects if Init not used or not enabled.
    Useful when user needs a code block inside Init
    to be executed with original layer init.
    """
    global _SHUTDOWN_INIT
    _SHUTDOWN_INIT = True


def restore_init_context():
    """
    Restore the model initializer.
    Suggestion: pair with shutdown_model_initializer
    """
    global _SHUTDOWN_INIT
    _SHUTDOWN_INIT = False


def _concat_partition(unit_tensors: List[Parameter], last_unit=False):

    # 0. Compute the unit numel
    unit_numel = sum(p.numel() for p in unit_tensors)

    # 1. Allocate memory if necessary
    if not is_flextrain_memory_allocated():
        allocate_flextrain_memory(unit_numel)
    memory = get_flextrain_memory()

    # 2. Check the unit numel
    if not last_unit:
        assert unit_numel == memory.unit_numel, \
            "Unit numel mismatch with allocated memory"

    # 3. Concatenate the whole unit into GPU contiguous memory
    temp_gpu_unit = memory.gpu_work_windows[0]
    tensor_group = ContiguousTensorGroup(unit_tensors, temp_gpu_unit)

    # 4. Append the tensor group to the memory coordinator
    coordinator._append_unit_tensor_group(tensor_group)

    # 5. Broadcast the whole unit from rank 0
    dist.broadcast(temp_gpu_unit, src=0)

    # 6. Partition the unit
    partitions = torch.chunk(temp_gpu_unit, dist.get_world_size())
    partition = partitions[dist.get_rank()]

    return partition


class Init(object):

    def __init__(self, layer_class: type, enabled=True):
        self._enabled = enabled
        self._layer_class = layer_class

        # Init related configurations
        self._layer_per_unit = get_flextrain_config().checkpoint_interval
        # self._para_ratio_on_devices = get_flextrain_config().para_ratio_on_devices

        # Track unit layers
        self._unit_layers: List[Module] = []

    def __enter__(self):
        if not self._enabled:
            return

        self._override_layer_init()

    def __exit__(self, *args, **kwargs):
        if not self._enabled:
            return

        # TODO: Finish potentially not finished layer init
        #       Because num_layer % checkpoint_interval can be non-zero

        self._restore_layer_init()

    def _override_layer_init(self):
        # Save the original layer init function
        self._original_layer_init = self._layer_class.__init__

        # Start of layer_init wrapper
        @functools.wraps(self._original_layer_init)
        def _concat_partition_init(module: Module, *args, **kwargs):

            # Conduct the original layer init function
            self._original_layer_init(module, *args, **kwargs)

            # If user wants to shutdown the model initializer
            # Then return immediately after original layer init
            if _SHUTDOWN_INIT:
                return

            # Track the current layer
            self._unit_layers.append(module)

            # If the current layer is not the last layer of the unit
            curr_layer = len(self._unit_layers)
            if curr_layer < self._layer_per_unit:
                return

            assert curr_layer == self._layer_per_unit

            #
            unit_tensors: List[Parameter] = []
            for layer in self._unit_layers:
                unit_tensors.extend(list(layer.parameters()))

            # Concatenate and partition the unit
            partition = _concat_partition(self._unit_layers)

            # Reset the unit layers
            self._unit_layers = []

        # End of layer_init wrapper

        self._layer_class.__init__ = _concat_partition_init

    def _restore_layer_init(self):
        self._layer_class.__init__ = self._original_layer_init


if __name__ == "__main__":
    # from torch import nn
    # from flextrain.config import init_flextrain_config
    # torch.manual_seed(0)
    # init_flextrain_config({
    #     "dtype": "fp16",
    #     "world_size": 1,
    #     "batch_size": 1,
    #     "micro_batch_size": 1,
    #     "micro_batch_per_block": 1,
    #     "num_layers": 3,
    #     "checkpoint_interval": 1
    # })
    # with Init(layer_class=nn.Linear):
    #     a = nn.Sequential(
    #         nn.Linear(10, 10, dtype=torch.float16),
    #         nn.Linear(10, 10, dtype=torch.float16),
    #         nn.Linear(10, 10, dtype=torch.float16)
    #     )
    # x = torch.randn(1, 10, dtype=torch.float16)
    # print(a(x))
    dist.init_distributed()

    a = torch.nn.Parameter(torch.randn(10, 10))
    b = torch.nn.Parameter(torch.randn(10, 10))
    # set_is_para_rank_unique_func(lambda x: x.data_ptr() == a.data_ptr())
    print_rank0(is_para_rank_unique(a))
    print_rank0(contiguous_allgathered_numel([a, b]))
    print_rank0(contiguous_partitioned_numel([a, b]))
