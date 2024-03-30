import torch
from torch.nn import Module

from deepspeed.runtime import DeepSpeedOptimizer
from deepspeed.runtime.flex.config import FLEX_CONFIG
from deepspeed.runtime.flex.scheduler import GreedySnakeBatchScheduler

from deepspeed.runtime.flex.checkpointing import (
    detach_variable,
    checkpointed_forward,
    retrieve_tensor_grads,
    checkpointed_backward
)

from typing import Sequence, Callable, Iterable, Any, Tuple


class FlexTrainFuncPack:
    """
    FlexTrainFuncPack is a class that contains the necessary
    functions for FlexTrainOptimizer to work properly.
    Users is responsible for providing relevant functions.

    Here are the functions that users need to provide:
    - get_layer_fn: returns the specified layer of the model
    - get_batch_fn: returns a batch of data when called
    - pre_process_fn: conducts the pre-processing before the forward pass
    - forward_fn: returns a forward function for the specified layers
    - post_process_fn: conducts the post-processing
    - loss_fn: calculates the loss
    """

    def __init__(
        self,
        model: Module,
        get_layer_fn: Callable[[Module, int], Module],
        get_batch_fn: Callable[[Iterable], Tuple[Tuple, Tuple, Tuple]],
        pre_process_fn: Callable[[Module, Tuple], Tuple[Tuple, Tuple]],
        forward_fn: Callable[[Module, int, int], Callable[[Tuple], Tuple]],
        post_process_fn: Callable[[Module, Tuple, Tuple], Tuple],
        loss_fn: Callable[[Tuple, Tuple], Sequence[Any]]
    ):
        self.model = model
        """
        The required model to be optimized by FlexTrainOptimizer.
        """

        self.get_layer_fn = get_layer_fn
        """
        get_layer_fn returns the specified layer of the model.
        Note that layers are indexed from 0.
        """

        self.get_batch_fn = get_batch_fn
        """
        get_batch_fn returns a batch of data when called.
        Batch is returned as a tuple of three tuples:
            model_input: inputs needed for the model forward pass
            post_input: inputs needed for the post processing
            loss_input: inputs needed for the loss calculation
        Note that data_iterator is passed as an argument every time.
        """

        self.pre_process_fn = pre_process_fn
        """
        pre_process_fn takes the model and model_input to conduct
        the pre-processing before the forward pass.
        Typically, token embedding is done in this function.
        pre_process_fn returns two tuples:
            passed_down: the input for the first layer
            each_layer: the input for each layer
        """

        self.forward_fn = forward_fn
        """
        forward_fn takes the model, layer start and end indices
        as inputs to return a forward function for the specified layers.
        The returned forward function is used as:
            passed_down = forward_fn(*passed_down, *each_layer)
        """

        self.post_process_fn = post_process_fn
        """
        post_process_fn takes the passed_down (provided by custom_forward_fn)
        and post_input as inputs to conduct the post-processing.
        The output is excatly the same as the model forward function.
        """

        self.loss_fn = loss_fn
        """
        loss_fn takes the llm_output (provided by post_process_fn)
        and loss_input as inputs to calculate the loss.
        It is an input-wrapper of the original loss function.
        It returns the standard scalar loss (the first element)
        and things that need to be stored (the second element).
        """

    def bind_data_iterator(self, data_iterator: Iterable):
        self.data_iterator = data_iterator

    def get_layer(self, layer_index: int):
        return self.get_layer_fn(self.model, layer_index)

    def get_batch(self):
        return self.get_batch_fn(self.data_iterator)

    def pre_process(self, model_input: Tuple):
        return self.pre_process_fn(self.model, model_input)

    def forward(self, layer_start: int, layer_end: int):
        return self.forward_fn(self.model, layer_start, layer_end)

    def post_process(self, passed_down: Tuple, post_input: Tuple):
        return self.post_process_fn(self.model, passed_down, post_input)

    def loss(self, llm_output: Tuple, loss_input: Tuple):
        return self.loss_fn(llm_output, loss_input)


class FlexTrainOptimizer(DeepSpeedOptimizer):
    """
    FlexTrainOptimizer is designed to maximize the training throughput.
    ZeRO Stage 3 optimization is used to support multi-GPU training.
    """

    def __init__(
        self,
        module: Module,
        init_optimizer: torch.optim.Optimizer
    ):
        super().__init__()

        # Take down the module, init_optimizer, and ft_config
        self.module = module
        self.init_optimizer = init_optimizer

        # Link to the FuncPack from module
        self.func_pack: FlexTrainFuncPack = module.func_pack

        # Create the GreedySnakeBatchScheduler
        self.scheduler = GreedySnakeBatchScheduler(
            FLEX_CONFIG.world_size,
            FLEX_CONFIG.batch_size,
            FLEX_CONFIG.micro_batch_size,
            FLEX_CONFIG.micro_batch_per_block,
            FLEX_CONFIG.num_layers,
            FLEX_CONFIG.checkpoint_interval
        )

    def forbackwards(self):
        ...


def hash_tensor(tensor):
    import torch
    import hashlib
    import numpy as np
    # Convert the tensor to a bytes object
    if isinstance(tensor, np.ndarray):
        tensor_bytes = tensor.tobytes()
    elif isinstance(tensor, torch.Tensor):
        try:
            tensor_bytes = tensor.cpu().numpy().tobytes()
        except BaseException:
            tensor_bytes = tensor.detach().cpu().numpy().tobytes()
    else:
        raise TypeError("tensor must be a numpy array or a torch tensor")

    # Calculate the SHA-256 hash
    sha256_hash = hashlib.sha256(tensor_bytes)

    # Return the hexadecimal representation of the hash
    return sha256_hash.hexdigest()

def test(model, data_iterator):
    model = model.cuda()
    model = model.to(dtype=torch.float16)
    func_pack: FlexTrainFuncPack = model.func_pack
    func_pack.bind_data_iterator(data_iterator)

    # with torch.no_grad():
    for i in range(10):
        model_input, post_input, loss_input = func_pack.get_batch()
        passed_down, each_layer = func_pack.pre_process(model_input)
        pre_out = passed_down

        ctxs = []
        for j in range(24):
            passed_down = detach_variable(passed_down)
            passed_down, ctx = checkpointed_forward(func_pack.forward(j, j + 1), *passed_down, *each_layer)
            ctxs.append(ctx)

        passed_down = detach_variable(passed_down)
        llm_output = func_pack.post_process(passed_down, post_input)
        loss, loss_store = func_pack.loss(llm_output, loss_input)

        # backward from loss to pre-post_process
        torch.autograd.backward(loss)

        # get the gradients
        passed_back = retrieve_tensor_grads(passed_down)
        passed_back += [None] * len(each_layer)

        ctxs.reverse()
        for ctx in ctxs:
            passed_back = checkpointed_backward(ctx, *passed_back)

        # do the backward of the pre-process
        torch.autograd.backward(pre_out[0], passed_back[0])

        assert False
