import torch

from typing import Iterable


def free_tensor(tensor: torch.Tensor):
    # Record the current stream if the tensor is on CUDA.
    if tensor.is_cuda:
        tensor.record_stream(torch.cuda.current_stream())

    # Reserve the device and dtype of the tensor.
    device = tensor.device
    dtype = tensor.dtype
    tensor.data = torch.empty(0, device=device, dtype=dtype)


def move_into_contiguous(src: Iterable[torch.Tensor], dst: torch.Tensor):

    offset = 0
    for tensor in src:
        # Get the number of elements in the tensor.
        numel = tensor.numel()

        # Get the shape of the tensor.
        shape = tensor.shape

        # Copy the tensor into the destination tensor.
        dst_target = dst[offset: offset + numel]
        dst_target.copy_(tensor.view(-1))

        # Free the memory of the source tensor.
        free_tensor(tensor)

        # Restore the original tensor.
        tensor.data = dst_target.view(shape)

        # Update the offset.
        offset += numel
