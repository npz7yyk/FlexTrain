from dataclasses import dataclass
from torch.nn import Module
from typing import Callable, Tuple, Sequence, Any


class LLMFunc:

    @staticmethod
    def get_layer(layer_index: int) -> Module:
        raise NotImplementedError

    @staticmethod
    def get_batch() -> Tuple[Tuple, Tuple, Tuple]:
        raise NotImplementedError

    @staticmethod
    def pre_process(pre_inputs: Tuple) -> Tuple[Tuple, Tuple]:
        raise NotImplementedError

    @staticmethod
    def layer_forward(
        start_layer_index: int,
        end_layer_index_plus1: int
    ) -> Callable[[Tuple, Tuple], Tuple]:
        raise NotImplementedError

    @staticmethod
    def post_process(passed_down: Tuple, post_inputs: Tuple) -> Tuple:
        raise NotImplementedError

    @staticmethod
    def loss(model_outputs: Tuple, loss_inputs: Tuple) -> Sequence[Any]:
        raise NotImplementedError


def set_llm_func(
    get_layer: Callable[[int], Module],
    get_batch: Callable[[], Tuple[Tuple, Tuple, Tuple]],
    pre_process: Callable[[Tuple], Tuple[Tuple, Tuple]],
    layer_forward: Callable[[int, int], Callable[[Tuple, Tuple], Tuple]],
    post_process: Callable[[Tuple, Tuple], Tuple],
    loss: Callable[[Tuple, Tuple], Sequence[Any]]
):
    LLMFunc.get_layer = get_layer
    LLMFunc.get_batch = get_batch
    LLMFunc.pre_process = pre_process
    LLMFunc.layer_forward = layer_forward
    LLMFunc.post_process = post_process
    LLMFunc.loss = loss
