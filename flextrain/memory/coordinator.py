import torch

from dataclasses import dataclass
from enum import Enum
from torch import Tensor
from torch.nn import Parameter

from flextrain.config import get_flextrain_config
from flextrain.model_initializer import LoopIndexer, FlexTrainMemory
from flextrain.utils import dist
from flextrain.utils.logging import rank0_logger


_MEMORY_COORDINATOR = None


def is_memory_coordinator_initialized():
    """
    Check if the memory coordinator is initialized.

    Returns:
        bool: True if the memory coordinator is initialized, False otherwise.
    """
    return _MEMORY_COORDINATOR is not None


def init_memory_coordinator(memory: FlexTrainMemory):
    """
    Initializes the memory coordinator.

    Args:
        memory (FlexTrainMemory): The allocated memory object.

    Returns:
        None

    Raises:
        AssertionError: If the memory coordinator is already initialized.
    """
    global _MEMORY_COORDINATOR
    assert _MEMORY_COORDINATOR is None, \
        "Memory coordinator already initialized"
    _MEMORY_COORDINATOR = FlexTrainMemoryCoordinator(memory)


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


class FlexTrainMemoryCoordinator:

    def __init__(self, memory: FlexTrainMemory):
        self.memory = memory

        self._map_layer_para = {}
        self._map_layer_grad = {}
        self._map_layer_opts = {}

        self._map_data_nvme_file = {}

    def _get_nvme_file(self, layer_index: int, data_type: FlexTrainDataTypes):
        return f"layer_{layer_index}_{data_type.name}"

    def _tensor_id(self, tensor: Tensor):
        return id(tensor)

    def _init_para(
        self,
        tensor: Tensor,
        layer_index: int
    ):
        assert layer_index not in self._map_layer_para

        # Register the tensor and the corresponding NVMe file
        self._map_layer_para[layer_index] = tensor
        self._map_data_nvme_file[self._tensor_id(tensor)] = \
            self._get_nvme_file(layer_index, FlexTrainDataTypes.PARA)
