import functools
import os
import torch

from torch import Tensor
from torch.nn import Parameter, Module
from typing import Callable, Iterable

from config import get_flextrain_config
from memory_coordinator import (
    is_memory_coordinator_initialized,
    init_memory_coordinator,
    get_memory_coordinator,
    FlexTrainMemoryCoordinator
)
from memory_utils import (
    move_into_contiguous
)


# Decorator of the layer initialization function
# This decorator is used to:
# 1. concatenate the parameters of the layer into a contiguous tensor
# 2. partition the contiguous tensor for dp and memory hierarchy
def concat_partition(layer_init_func: Callable):

    @functools.wraps(layer_init_func)
    def wrapper(module: Module, *args, **kwargs):
        layer_init_func(module, *args, **kwargs)

        if not is_memory_coordinator_initialized():
            layer_numel = sum(p.numel() for p in module.parameters())
            init_memory_coordinator(layer_numel)

        layer_bases = get_memory_coordinator().cpu_layer_bases
        move_into_contiguous(module.parameters(), layer_bases[0])
        layer_bases.step()

    return wrapper


# This context manager class is modified from
# class InsertPostInitMethodToModuleSubClasses in
# deepspeed/runtime/zero/partition_parameters.py
class Init(object):

    def __init__(self, layer_class: type, enabled=True):
        self.enabled = enabled
        self.layer_class = layer_class

        self.curr_layer = 0

    def __enter__(self):
        if not self.enabled:
            return

        self._override_layer_init(self.layer_class)

    def __exit__(self, *args, **kwargs):
        if not self.enabled:
            return

        self._restore_layer_init(self.layer_class)

    def _override_layer_init(self, layer_class):
        self._original_layer_init = layer_class.__init__
        layer_class.__init__ = concat_partition(layer_class.__init__)

    def _restore_layer_init(self, layer_class):
        layer_class.__init__ = self._original_layer_init


if __name__ == "__main__":
    from torch import nn
    from config import init_flextrain_config
    torch.manual_seed(0)
    init_flextrain_config({
        "dtype": "fp16",
        "world_size": 1,
        "batch_size": 1,
        "micro_batch_size": 1,
        "micro_batch_per_block": 1,
        "num_layers": 3,
        "checkpoint_interval": 1
    })
    with Init(layer_class=nn.Linear):
        a = nn.Sequential(
            nn.Linear(10, 10, dtype=torch.float16),
            nn.Linear(10, 10, dtype=torch.float16),
            nn.Linear(10, 10, dtype=torch.float16)
        )
    x = torch.randn(1, 10, dtype=torch.float16)
    print(a(x))
