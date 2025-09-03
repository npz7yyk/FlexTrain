import os
import torch

from torch.distributed import ReduceOp
import torch.distributed

from flextrain.defaults import (
    PROCESS_GROUP_TIMEOUT_DEFAULT,
    TORCH_DISTRIBUTED_BACKEND_DEFAULT,
    TORCH_DISTRIBUTED_INIT_METHOD_DEFAULT
)
from flextrain.utils.logging import rank0_logger
from flextrain.utils.torch import (
    get_all_gather_function,
    get_reduce_scatter_function
)

_ALL_GATHER_FUNCTION = None
_REDUCE_SCATTER_FUNCTION = None

_INITIALIZED = False
_LOCAL_RANK = 0
_WORLD_SIZE = 1


def init_distributed(
    dist_backend=TORCH_DISTRIBUTED_BACKEND_DEFAULT,
    timeout=PROCESS_GROUP_TIMEOUT_DEFAULT,
    init_method=TORCH_DISTRIBUTED_INIT_METHOD_DEFAULT,
    force_init=True,
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
        force_init: Optional (bool). Default is True.
            If False, dist initialization will be skipped for single-GPU mode.
        rank: Optional (int). Default is -1.
            The current manually specified rank. Needed by some init_methods.
        world_size: Optional (int). Default is -1.
            Desired world_size for the TCP / Shared file-system initialization.
    """
    global _INITIALIZED

    # If distributed is already initialized, skip initialization
    if _INITIALIZED:
        return

    _INITIALIZED = True

    if int(os.getenv("WORLD_SIZE", "0")) == 0:
        rank0_logger.info(
            "Environment variable WORLD_SIZE is not set. "
            "Defaulting to single-GPU training. (WORLD_SIZE=1)"
        )
        os.environ["WORLD_SIZE"] = "1"

    if int(os.getenv("WORLD_SIZE", "0")) == 1 and not force_init:
        rank0_logger.info(
            "FlexTrain is running in single-GPU mode. "
            "Distributed backend initialization is skipped."
        )
        return

    global _ALL_GATHER_FUNCTION, _REDUCE_SCATTER_FUNCTION

    _ALL_GATHER_FUNCTION = get_all_gather_function()
    _REDUCE_SCATTER_FUNCTION = get_reduce_scatter_function()

    # Get the number of CPU threads available
    n_cpu_threads = os.cpu_count()
    n_cpu_threads = max(1, n_cpu_threads // int(os.getenv("WORLD_SIZE")))
    # Set the number of CPU threads to use for distributed training
    # torch.distributed will set OMP_NUM_THREADS to 1 if not set
    if int(os.getenv("OMP_NUM_THREADS", "0")) <= n_cpu_threads:
        os.environ["OMP_NUM_THREADS"] = str(n_cpu_threads)
    if torch.get_num_threads() <= n_cpu_threads:
        torch.set_num_threads(n_cpu_threads)

    rank0_logger.info(
        f"\n\n> FlexTrain initializing backend {dist_backend}\n"
        f"  - Number of processes: {os.getenv('WORLD_SIZE')}\n"
        f"  - OMP_NUM_THREADS per process: {os.getenv('OMP_NUM_THREADS')}\n"
    )

    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            dist_backend,
            timeout=timeout,
            init_method=init_method,
            rank=rank,
            world_size=world_size
        )

    # FlexTrain currently only supports single-node training
    global _LOCAL_RANK, _WORLD_SIZE
    _LOCAL_RANK = torch.distributed.get_rank()
    _WORLD_SIZE = torch.distributed.get_world_size()

    # Set the device to the current rank
    torch.cuda.set_device(_LOCAL_RANK)


def _info_not_initialized():
    if not _INITIALIZED:
        rank0_logger.info(
            "FlexTrain distributed is not explicitly initialized. "
            "Trying to initialize it automatically ..."
        )
        init_distributed(force_init=False)


def get_rank():
    _info_not_initialized()
    return _LOCAL_RANK


def get_world_size():
    _info_not_initialized()
    return _WORLD_SIZE


def is_single_process():
    _info_not_initialized()
    return _WORLD_SIZE == 1


class DummyHandle:
    """ Handle that imitates a torch.distributed handle but does nothing. """

    def __init__(self, *args, **kwargs):
        pass

    def wait(self):
        pass

    def is_completed(self):
        return True


def barrier(group=None):
    _info_not_initialized()
    if is_single_process():
        return DummyHandle()
    return torch.distributed.barrier(group)


def print_rank0(*args, **kwargs):
    if get_rank() == 0:
        print(*args, **kwargs)


def print_rank_by_rank(*args, **kwargs):
    for i in range(get_world_size()):
        barrier()
        if i == get_rank():
            print(*args, **kwargs)


def broadcast(tensor, src, group=None, async_op=False):
    _info_not_initialized()
    if is_single_process():
        return DummyHandle()
    return torch.distributed.broadcast(tensor, src, group, async_op)


def all_gather(tensor_out, tensor_in, group=None, async_op=False):
    _info_not_initialized()
    if is_single_process():
        return DummyHandle()
    return _ALL_GATHER_FUNCTION(tensor_out, tensor_in, group, async_op)


def all_reduce(tensor, op=ReduceOp.SUM, group=None, async_op=False):
    _info_not_initialized()
    if is_single_process():
        return DummyHandle()
    return torch.distributed.all_reduce(tensor, op, group, async_op)


def reduce_scatter(
    tensor_out, tensor_in,
    op=ReduceOp.SUM, group=None, async_op=False
):
    _info_not_initialized()
    if is_single_process():
        return DummyHandle()
    return _REDUCE_SCATTER_FUNCTION(tensor_out, tensor_in, op, group, async_op)
