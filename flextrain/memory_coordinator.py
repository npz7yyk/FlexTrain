import torch

from typing import SupportsIndex

from .config import get_flextrain_config


class LoopIndexer:
    """
    LoopIndexer is a class that wraps a target that supports index.
    It provides a way to loop through the target in a circular manner.
    """

    def __init__(self, target: SupportsIndex):
        self.target = target
        self.total_items = len(target)

        self.index_offset = 0

    def step(self):
        self.index_offset += 1
        if self.index_offset >= self.total_items:
            self.index_offset = 0

    def __getitem__(self, index: int):
        curr_index = (index + self.index_offset) % self.total_items
        return self.target[curr_index]


class FlexTrainMemoryCoordinator:

    def __init__(self, layer_numel: int):
        config = get_flextrain_config()

        self.dtype = config.dtype
        self.num_layers = config.num_layers
        self.layer_numel = layer_numel

        # allocate memory
        self._allocate_cuda_memory()
        self._allocate_cpu_memory()
        pass

    def _allocate_cuda_memory(self):
        torch.cuda.empty_cache()

        layer_numel = self.layer_numel
        work_window_mem = torch.empty(
            self.layer_numel * 1,
            dtype=self.dtype,
            device=torch.cuda.current_device()
        )

        work_windows = []
        for i in range(1):
            window = work_window_mem[i * layer_numel: (i + 1) * layer_numel]
            work_windows.append(window)

        self.work_windows = LoopIndexer(work_windows)

    def _allocate_cpu_memory(self):
        num_layers = self.num_layers
        layer_numel = self.layer_numel

        cpu_base_mem = torch.empty(
            num_layers * layer_numel,
            dtype=torch.float16,
            device=torch.device('cpu'),
            pin_memory=True
        )

        cpu_layer_bases = []
        for i in range(num_layers):
            base = cpu_base_mem[i * layer_numel: (i + 1) * layer_numel]
            cpu_layer_bases.append(base)

        self.cpu_layer_bases = LoopIndexer(cpu_layer_bases)


_MEMORY_COORDINATOR = None


def is_memory_coordinator_initialized():
    """
    Check if the memory coordinator is initialized.

    Returns:
        bool: True if the memory coordinator is initialized, False otherwise.
    """
    return _MEMORY_COORDINATOR is not None


def init_memory_coordinator(layer_numel: int):
    """
    Initializes the memory coordinator.

    Args:
        layer_numel (int): The number of elements in each layer.

    Returns:
        None

    Raises:
        AssertionError: If the memory coordinator is already initialized.
    """
    global _MEMORY_COORDINATOR
    assert _MEMORY_COORDINATOR is None, \
        "Memory coordinator already initialized"
    _MEMORY_COORDINATOR = FlexTrainMemoryCoordinator(layer_numel)


def get_memory_coordinator():
    """
    Get the memory coordinator.

    Returns:
        FlexTrainMemoryCoordinator: The memory coordinator.

    Raises:
        AssertionError: If the memory coordinator is not initialized.
    """
    assert _MEMORY_COORDINATOR is not None, \
        "Memory coordinator not initialized"
    return _MEMORY_COORDINATOR
