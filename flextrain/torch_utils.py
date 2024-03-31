import torch

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


def has_coalescing_manager():
    has_c10d = hasattr(torch.distributed, 'distributed_c10d')
    if not has_c10d:
        return False
    return hasattr(torch.distributed.distributed_c10d, '_coalescing_manager')


def has_all_reduce_coalesced():
    return hasattr(torch.distributed, "all_reduce_coalesced") \
        and torch_version_ge("1.13")


def get_coalescing_manager(group, device, reqs, async_op):
    build_func = torch.distributed.distributed_c10d._coalescing_manager
    if torch_version_eq("2.0"):
        return build_func(group, device=device, reqs=reqs)
    elif torch_version_ge("2.1"):
        return build_func(group, device=device, async_ops=async_op)
    else:
        return build_func(group, reqs)


def get_all_gather_function():
    if hasattr(torch.distributed, "all_gather_into_tensor"):
        return torch.distributed.all_gather_into_tensor
    elif hasattr(torch.distributed, "_all_gather_base"):
        return torch.distributed._all_gather_base
    return None


def get_reduce_scatter_function(self):
    if hasattr(torch.distributed, "reduce_scatter_tensor"):
        return torch.distributed.reduce_scatter_tensor
    elif hasattr(torch.distributed, "_reduce_scatter_base"):
        return torch.distributed._reduce_scatter_base
    return None
