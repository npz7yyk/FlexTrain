from dataclasses import dataclass
from torch.nn import Module
from typing import Callable, Tuple, Sequence, Any


@dataclass
class LLMFuncPack:
    get_layer: Callable[[int], Module]
    get_batch: Callable[[], Tuple[Tuple, Tuple, Tuple]]
    pre_process: Callable[[Tuple], Tuple[Tuple, Tuple]]
    layer_forward: Callable[[int, int], Callable[[Tuple], Tuple]]
    post_process: Callable[[Tuple, Tuple], Tuple]
    loss: Callable[[Tuple, Tuple], Sequence[Any]]
