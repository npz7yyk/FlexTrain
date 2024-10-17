import torch

from enum import Enum
from torch import Tensor
from torch.nn import Parameter
from typing import Iterable, List

from flextrain.config import get_flextrain_config
from flextrain.utils.distributed import get_rank


class FlexTrainDataType(Enum):
    """
    Enum for the data type used in FlexTrain.
    """
    CKPT = 0
    PARA = 1
    GRAD = 2
    OPTS = 3


class FlexTrainDataID:

    def __init__(
        self,
        unit_index: int,
        data_type: FlexTrainDataType
    ):
        self.unit_index = unit_index
        self.data_type = data_type

    def __str__(self):
        checkpoint_interval = get_flextrain_config().checkpoint_interval
        start = self.unit_index * checkpoint_interval
        end = start + checkpoint_interval - 1
        layer_str = f"layer{start}-{end}" \
            if checkpoint_interval > 1 else f"layer{start}"
        return (
            f"rank{get_rank()}_"
            f"{layer_str}_"
            f"{self.data_type.name}.swp"
        )


def align_numel(original_numel: int, align_size: int):
    return (original_numel + align_size - 1) // align_size * align_size


def free_tensor(tensor: Tensor, record_stream=False):
    # Record the current stream if the tensor is on CUDA.
    if tensor.is_cuda and record_stream:
        tensor.record_stream(torch.cuda.current_stream())

    # Reserve the device and dtype of the tensor.
    device = tensor.device
    dtype = tensor.dtype
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


def copy_segment(
    srcs: List[torch.Tensor],
    dsts: List[torch.Tensor],
    force_equal: bool = True
) -> None:
    """ Copies from a list of source tensors to a list of destination tensors.

    Args:
        srcs (List[torch.Tensor]): List of source tensors.
        dsts (List[torch.Tensor]): List of destination tensors.
        force_equal (bool, optional): \
            Whether to force the total number of elements in the source \
            and destination tensors to be equal. Defaults to True.
    """
    srcs = [src.flatten() for src in srcs if src.numel() > 0]
    dsts = [dst.flatten() for dst in dsts if dst.numel() > 0]

    total_src_elements = sum(src.numel() for src in srcs)
    total_dst_elements = sum(dst.numel() for dst in dsts)

    if force_equal:
        assert total_src_elements == total_dst_elements

    inner_src_offset = 0
    inner_dst_offset = 0

    while srcs and dsts:
        src = srcs[0]
        dst = dsts[0]

        src_numel = src.numel() - inner_src_offset
        dst_numel = dst.numel() - inner_dst_offset

        numel_to_copy = min(src_numel, dst_numel)

        src_seg = src[inner_src_offset: inner_src_offset + numel_to_copy]
        dst_seg = dst[inner_dst_offset: inner_dst_offset + numel_to_copy]

        dst_seg.copy_(src_seg, non_blocking=True)

        inner_src_offset += numel_to_copy
        inner_dst_offset += numel_to_copy

        if inner_src_offset >= src.numel():
            inner_src_offset = 0
            srcs.pop(0)
        if inner_dst_offset >= dst.numel():
            inner_dst_offset = 0
            dsts.pop(0)


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
            # If p.grad is None, create a new tensor.
            if p.grad is None:
                p.grad = contiguous_mem[offset: offset + n].view(s)
            else:
                p.grad.data = contiguous_mem[offset: offset + n].view(s)
            offset += n

    def detach_para(self):
        for para in self.paras:
            free_tensor(para)

    def detach_grad(self):
        for para in self.paras:
            free_tensor(para.grad)


class Waitable:
    def __init__(self):
        self._finished = False

    def is_completed(self):
        return self._finished

    def wait(self):
        if self._finished:
            return
        self._wait_task()
        self._finished = True

    def _wait_task(self):
        # Implement this method in the subclass.
        ...


class DummyHandle(Waitable):
    def _wait_task(self):
        pass


class FunctionHandle(Waitable):
    def __init__(self, function, *args, **kwargs):
        super().__init__()
        self._function = function
        self._args = args
        self._kwargs = kwargs

    def _wait_task(self):
        self._function(*self._args, **self._kwargs)


class AsyncIOHandle(Waitable):
    def __init__(self, handle: Waitable):
        super().__init__()
        self._handle = handle

    def _wait_task(self):
        self._handle.wait()


class FusedHandle(Waitable):
    def __init__(self, handles: Iterable[Waitable]):
        super().__init__()
        self._handles = list(handles)

    def _wait_task(self):
        for handle in self._handles:
            handle.wait()
