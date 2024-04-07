import torch
import torch.distributed as dist

from torch.distributed import ReduceOp

from flextrain.config import get_flextrain_config
from flextrain.defaults import (
    PROCESS_GROUP_TIMEOUT_DEFAULT,
    TORCH_DISTRIBUTED_BACKEND_DEFAULT,
    TORCH_DISTRIBUTED_INIT_METHOD_DEFAULT
)
from flextrain.utils.logging import rank0_logger
from flextrain.utils.torch import (
    get_all_gather_function,
    get_coalescing_manager,
    get_reduce_scatter_function
)

_ALL_GATHER_FUNCTION = None
_REDUCE_SCATTER_FUNCTION = None

_INITIALIZED = False
_LOCAL_RANK = 0
_WORLD_SIZE = 1


def _warn_not_initialized():
    if not _INITIALIZED:
        rank0_logger.warning_once(
            "FlexTrain distributed is not explicitly initialized. "
            "Defaulting to single-node training ..."
            ""
        )


def _assert_torch_distributed_initialized():
    assert dist.is_initialized(), "flextrain.distributed is not initialized"


def init_distributed(
    dist_backend=TORCH_DISTRIBUTED_BACKEND_DEFAULT,
    timeout=PROCESS_GROUP_TIMEOUT_DEFAULT,
    init_method=TORCH_DISTRIBUTED_INIT_METHOD_DEFAULT,
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
    if dist.is_initialized():
        return

    global _INITIALIZED
    _INITIALIZED = True

    if get_flextrain_config().world_size == 1:
        rank0_logger.info(
            "FlexTrain is running in single-node mode. "
            "Distributed backend initialization is skipped."
        )
        return

    global _ALL_GATHER_FUNCTION, _REDUCE_SCATTER_FUNCTION

    _ALL_GATHER_FUNCTION = get_all_gather_function()
    _REDUCE_SCATTER_FUNCTION = get_reduce_scatter_function()

    rank0_logger.info(f"FlexTrain initializing backend {dist_backend}")

    dist.init_process_group(
        dist_backend,
        timeout=timeout,
        init_method=init_method,
        rank=rank,
        world_size=world_size
    )

    # FlexTrain currently only supports single-node training
    global _LOCAL_RANK, _WORLD_SIZE
    _LOCAL_RANK = dist.get_rank()
    _WORLD_SIZE = dist.get_world_size()

    # Set the device to the current rank
    torch.cuda.set_device(_LOCAL_RANK)


def get_rank():
    _warn_not_initialized()
    return _LOCAL_RANK


def get_world_size():
    _warn_not_initialized()
    return _WORLD_SIZE


def current_device():
    _warn_not_initialized()
    return _LOCAL_RANK


def barrier(group=None):
    _assert_torch_distributed_initialized()
    return dist.barrier(group)


def broadcast(tensor, src, group=None, async_op=False):
    _assert_torch_distributed_initialized()
    return dist.broadcast(tensor, src, group, async_op)


def all_gather(tensor_list, tensor, group=None, async_op=False):
    _assert_torch_distributed_initialized()
    return _ALL_GATHER_FUNCTION(tensor_list, tensor, group, async_op)


def all_gather_coalesced(output_tensor_list, input_tensor, group=None, async_op=False):
    _assert_torch_distributed_initialized()
    return get_coalescing_manager().all_gather_coalesced(
        output_tensor_list, input_tensor, group, async_op
    )


def all_reduce(tensor, op=ReduceOp.SUM, group=None, async_op=False):
    _assert_torch_distributed_initialized()
    return dist.all_reduce(tensor, op, group, async_op)
