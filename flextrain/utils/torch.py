import torch
import torch.distributed as dist

TORCH_MAJOR, TORCH_MINOR = map(int, torch.__version__.split('.')[:2])


def torch_version_eq(cmp_version: str):
    DESIRE_MAJOR, DESIRE_MINOR = map(int, cmp_version.split('.')[:2])
    return TORCH_MAJOR == DESIRE_MAJOR and TORCH_MINOR == DESIRE_MINOR


def torch_version_ge(cmp_version: str):
    DESIRE_MAJOR, DESIRE_MINOR = map(int, cmp_version.split('.')[:2])
    if TORCH_MAJOR > DESIRE_MAJOR:
        return True
    elif TORCH_MAJOR == DESIRE_MAJOR:
        return TORCH_MINOR >= DESIRE_MINOR
    else:
        return False


def _has_coalescing_manager():
    has_c10d = hasattr(dist, 'distributed_c10d')
    if not has_c10d:
        return False
    return hasattr(dist.distributed_c10d, '_coalescing_manager')


def _has_all_reduce_coalesced():
    return hasattr(dist, "all_reduce_coalesced") \
        and torch_version_ge("1.13")


assert _has_coalescing_manager(), \
    "Current torch version does not have all_reduce_coalesced api"
assert _has_all_reduce_coalesced(), \
    "Current torch version does not have all_reduce_coalesced api"


def get_coalescing_manager(group, device, reqs, async_op):
    build_func = dist.distributed_c10d._coalescing_manager
    if torch_version_eq("2.0"):
        return build_func(group, device=device, reqs=reqs)
    elif torch_version_ge("2.1"):
        return build_func(group, device=device, async_ops=async_op)
    else:
        return build_func(group, reqs)


def get_all_gather_function():
    if hasattr(dist, "all_gather_into_tensor"):
        return dist.all_gather_into_tensor
    elif hasattr(dist, "_all_gather_base"):
        return dist._all_gather_base
    return None


def get_reduce_scatter_function():
    if hasattr(dist, "reduce_scatter_tensor"):
        return dist.reduce_scatter_tensor
    elif hasattr(dist, "_reduce_scatter_base"):
        return dist._reduce_scatter_base
    return None
