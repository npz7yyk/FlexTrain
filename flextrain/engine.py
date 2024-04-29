import torch

from torch.nn import Module

from flextrain import distributed as dist
from flextrain.checkpointing import (
    detach_variable,
    checkpointed_forward,
    checkpointed_backward,
    retrieve_tensor_grads
)
from flextrain.config import get_flextrain_config
from flextrain.llm_func import LLMFuncPack
from flextrain.memory.coordinator import get_memory_coordinator
from flextrain.scheduler import GreedySnakeBatchScheduler


class FlexTrainEngine(object):
    """
    FlexTrainEngine is designed to maximize the training throughput.
    ZeRO Stage 3 optimization is used to support multi-GPU training.
    """

    def __init__(
        self,
        module: Module,
        init_optimizer: torch.optim.Optimizer,
        llm_functions: LLMFuncPack
    ):
        super().__init__()

        # Get FlexTrain configuration
        config = get_flextrain_config()

        # Link to the model and optimizer
        self.module = module.cuda().to(dtype=config.device_dtype)
        self.init_optimizer = init_optimizer

        # Link to LLM function pack
        self.llm_funcs: LLMFuncPack = llm_functions

        # Link to the memory coordinator
        self.memory_coordinator = get_memory_coordinator()

        # Create the GreedySnakeBatchScheduler
        self.scheduler = GreedySnakeBatchScheduler(
            dist.get_world_size(),
            config.batch_size,
            config.micro_batch_size,
            config.micro_batch_per_block,
            config.num_layers,
            config.checkpoint_interval
        )

    def train(self):
        self.module.train()

    def eval(self):
        self.module.eval()

    @torch.no_grad()
    def forward_backward(self):
        llm_funcs = self.llm_funcs

        for i in range(3):
            pre_input, post_input, loss_input = llm_funcs.get_batch()

            self.memory_coordinator.warmup_forward_pipeline()
            passed_down, each_layer = llm_funcs.pre_process(pre_input)

            ctxs = []
            for j in range(24):
                passed_down = detach_variable(passed_down)
                self.memory_coordinator.pre_forward_unit(j)
                passed_down, ctx = checkpointed_forward(
                    llm_funcs.layer_forward(j, j + 1),
                    passed_down, each_layer
                )
                self.memory_coordinator.post_forward_unit(j)
                ctxs.append(ctx)

            passed_down = detach_variable(passed_down)
            llm_output = llm_funcs.post_process(passed_down, post_input)
            loss, loss_store = llm_funcs.loss(llm_output, loss_input)

            for w in range(dist.get_world_size()):
                dist.barrier()
                if dist.get_rank() == w:
                    print(loss)
                    if w == dist.get_world_size() - 1:
                        print()
                dist.barrier()
        assert False


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


def test(model):
    model = model.to(dtype=torch.float16)
    model = model.cuda()
    func_pack = get_llm_func_pack()

    # with torch.no_grad():
    for i in range(10):
        pre_input, post_input, loss_input = func_pack.get_batch()
        passed_down, each_layer = func_pack.pre_process(pre_input)

        ctxs = []
        for j in range(24):
            passed_down = detach_variable(passed_down)
            passed_down, ctx = checkpointed_forward(
                func_pack.forward(j, j + 1),
                passed_down, each_layer
            )
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
        # torch.autograd.backward(pre_out[0], passed_back[0])

        assert False
