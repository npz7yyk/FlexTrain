import os
import torch
import torch.distributed as dist

from torch.distributed import ReduceOp

from flextrain.defaults import (
    DEFAULT_PROCESS_GROUP_TIMEOUT,
    DEFAULT_TORCH_DISTRIBUTED_BACKEND,
    DEFAULT_TORCH_DISTRIBUTED_INIT_METHOD
)
from .logging import logger
from .torch import (
    get_all_gather_function,
    get_coalescing_manager,
    get_reduce_scatter_function
)

_ALL_GATHER_FUNCTION = None
_REDUCE_SCATTER_FUNCTION = None


def is_distributed_intialized():
    return dist.is_initialized()


_WORLD_RANK = -1
_WORLD_SIZE = -1


def init_distributed(
    dist_backend=DEFAULT_TORCH_DISTRIBUTED_BACKEND,
    timeout=DEFAULT_PROCESS_GROUP_TIMEOUT,
    init_method=DEFAULT_TORCH_DISTRIBUTED_INIT_METHOD,
    rank=-1,
    world_size=-1
):
    """
    Initialize dist backend, potentially performing MPI discovery if needed

    Arguments:
        dist_backend: Optional (str). Default is nccl.
            torch distributed backend, e.g., nccl, mpi, gloo
        timeout: Optional (timedelta). Default value equals 30 minutes.
            Timeout for operations executed against the process group.
        init_method: Optional (string). Default is “env://”.
            URL specifying how to initialize the process group.
        rank: Optional (int). Default is -1.
            The current manually specified rank. Needed by some init_methods.
        world_size: Optional (int). Default is -1.
            Desired world_size for the TCP / Shared file-system initialization.
    """
    if is_distributed_intialized():
        return

    logger.info(f"FlexTrain initializing backend {dist_backend}")

    global _ALL_GATHER_FUNCTION, _REDUCE_SCATTER_FUNCTION

    _ALL_GATHER_FUNCTION = get_all_gather_function()
    _REDUCE_SCATTER_FUNCTION = get_reduce_scatter_function()

    if dist.is_initialized():
        return

    dist.init_process_group(
        dist_backend,
        timeout=timeout,
        init_method=init_method,
        rank=rank,
        world_size=world_size
    )

    # FlexTrain currently only supports single-node training
    global _WORLD_RANK, _WORLD_SIZE
    _WORLD_RANK = dist.get_rank()
    _WORLD_SIZE = dist.get_world_size()


def get_rank():
    assert is_distributed_intialized(), "Distributed is not initialized"
    return _WORLD_RANK


def get_world_size():
    assert is_distributed_intialized(), "Distributed is not initialized"
    return _WORLD_SIZE


def current_device():
    assert is_distributed_intialized(), "Distributed is not initialized"
    return _WORLD_RANK


def all_reduce(tensor, op=ReduceOp.SUM, group=None, async_op=False):
    assert is_distributed_intialized(), "Distributed is not initialized"
    return dist.all_reduce(tensor, op, group, async_op)
