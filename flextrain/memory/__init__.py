import torch

from enum import Enum
from torch import Tensor
from torch.nn import Parameter
from typing import Iterable

from flextrain.config import get_flextrain_config
from flextrain.utils.distributed import get_rank, get_world_size


class FlexTrainDataTypes(Enum):
    """
    Enum for the data types used in FlexTrain.
    """
    CKPT = 0
    PARA = 1
    GRAD = 2
    OPTS = 3


class FlexTrainDataID:

    def __init__(
        self,
        unit_index: int,
        data_type: FlexTrainDataTypes
    ):
        self.unit_index = unit_index
        self.data_type = data_type

    def __str__(self):
        checkpoint_interval = get_flextrain_config().checkpoint_interval
        num_layers = get_flextrain_config().num_layers
        start = self.unit_index * checkpoint_interval
        end = start + checkpoint_interval - 1
        end = min(end, num_layers - 1)
        if start == end:
            layer_str = f"layer{start}"
        else:
            layer_str = f"layer{start}-{end}"
        return (
            f"rank{get_rank()}_"
            f"{layer_str}_"
            f"{self.data_type.name}.swp"
        )


def contiguous_allgathered_numel(paras: Iterable[Parameter]):
    """ Allgathered numel of the contiguous memory for the parameters. """

    world_size = get_world_size()
    total_numel = sum(para.numel() for para in paras)

    # Align the total numel to the world size.
    if total_numel % world_size:
        total_numel += world_size - total_numel % world_size

    return total_numel


def contiguous_partitioned_numel(paras: Iterable[Parameter]):
    """ Partitioned numel of the contiguous memory for the parameters. """

    world_size = get_world_size()
    total_numel = sum(para.numel() for para in paras)

    # Align the total numel to the world size.
    if total_numel % world_size:
        total_numel += world_size - total_numel % world_size

    return total_numel // world_size


def free_tensor(tensor: Tensor, set_none=False):
    # Record the current stream if the tensor is on CUDA.
    if tensor.is_cuda:
        tensor.record_stream(torch.cuda.current_stream())

    # Reserve the device and dtype of the tensor.
    device = tensor.device
    dtype = tensor.dtype
    if set_none:
        tensor.data = None
    else:
        tensor.data = torch.empty(0, device=device, dtype=dtype)


def move_into_contiguous(srcs: Iterable[Tensor], dst: Tensor):

    offset = 0
    for tensor in srcs:
        # Get the number of elements in the tensor.
        numel = tensor.numel()

        # Get the shape of the tensor.
        shape = tensor.shape

        # Copy the tensor into the destination tensor.
        dst_target = dst[offset: offset + numel]
        dst_target.copy_(tensor.data.view(-1))

        # Free the memory of the source tensor.
        free_tensor(tensor)

        # Restore the original tensor.
        tensor.data = dst_target.view(shape)

        # Update the offset.
        offset += numel


class ContiguousParaGroup:
    """
    Manage a group of parameters in contiguous memory.

    Args:
        paras: Iterable[Parameter]: The parameters to be managed.
    """
    def __init__(self, paras: Iterable[Parameter]):
        self.paras = list(paras)
        self.numels = [para.numel() for para in self.paras]
        self.shapes = [para.shape for para in self.paras]

    def link_para_to(self, contiguous_mem: Tensor):
        offset = 0
        for p, n, s in zip(self.paras, self.numels, self.shapes):
            p.data = contiguous_mem[offset: offset + n].view(s)
            offset += n

    def link_grad_to(self, contiguous_mem: Tensor):
        offset = 0
        for p, n, s in zip(self.paras, self.numels, self.shapes):
            p.grad.data = contiguous_mem[offset: offset + n].view(s)
            offset += n

    def detach_para(self):
        for para in self.paras:
            free_tensor(para)

    def detach_grad(self):
        for para in self.paras:
            free_tensor(para.grad, set_none=True)
