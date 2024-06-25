import torch

from torch.nn import Module
from typing import Any, Sequence, Tuple, List, Dict

from flextrain import distributed as dist
from flextrain.checkpointing import (
    FWDContext,
    detach_variable,
    checkpointed_forward,
    checkpointed_backward,
    retrieve_tensor_grads
)
from flextrain.config import get_flextrain_config
from flextrain.llm_func import LLMFunc, retrieve_llm_loss
from flextrain.memory.coordinator import get_model_coordinator
from flextrain.scheduler import LLMTask, GreedySnakeBlockScheduler


class FlexTrainEngine(object):
    """
    FlexTrainEngine is designed to maximize the training throughput.
    ZeRO Stage 3 optimization is used to support multi-GPU training.
    """

    def __init__(
        self,
        module: Module,
        init_optimizer: torch.optim.Optimizer
    ):
        super().__init__()

        # Get FlexTrain configuration
        config = get_flextrain_config()

        # Link to the model and optimizer
        self.module = module.cuda().to(dtype=config.device_dtype)
        self.init_optimizer = init_optimizer

        # Link to the model coordinator
        self.model_coordinator = get_model_coordinator()

        # LLM training information
        self.micro_batch_per_block = self._compute_micro_batch_per_block()
        self.num_units = self.model_coordinator.num_units

        # Create the scheduler
        self.scheduler = GreedySnakeBlockScheduler(
            self.micro_batch_per_block,
            self.num_units
        )

        # Create context containers
        self._reset_context()

    def _compute_micro_batch_per_block(self):
        config = get_flextrain_config()
        batch_size = config.batch_size
        micro_batch_size = config.micro_batch_size
        world_size = dist.get_world_size()
        assert batch_size % (micro_batch_size * world_size) == 0, \
            "Batch size must be divisible by world size x micro_batch_size"
        return batch_size // (micro_batch_size * world_size)

    def _reset_context(self):
        # Recent passed down container
        self.micro_batch_passed_down: Dict[int, Tuple] = {}
        # Recent passed back container
        self.micro_batch_passed_back: Dict[int, Tuple] = {}

        # Inputs container
        self.micro_batch_post_inputs: Dict[int, Tuple] = {}
        self.micro_batch_loss_inputs: Dict[int, Tuple] = {}
        self.micro_batch_every_layer: Dict[int, Tuple] = {}
        self.every_layer_len = 0

        # Context container
        self.micro_batch_unit_ctx: List[List[FWDContext]] = [
            [None] * self.num_units
            for _ in range(self.micro_batch_per_block)
        ]

        # Loss result container
        self.micro_batch_loss_rsts: Dict[int, Sequence[Any]] = {}

    def train(self):
        self.module.train()

    def eval(self):
        self.module.eval()

    def train_iteration(self):
        # 1. Conduct all tasks assigned by the scheduler
        for task in self.scheduler:
            if task.is_forwarding:
                self._conduct_forward(task)
            else:
                self._conduct_backward(task)
        # 2. Conduct reset
        self._reset_context()

    def _batch_pre_process(self):
        for i in range(self.micro_batch_per_block):
            pre_inputs, post_inputs, loss_inputs = LLMFunc.get_batch()
            # Store the inputs, no need to offload from GPU
            self.micro_batch_post_inputs[i] = post_inputs
            self.micro_batch_loss_inputs[i] = loss_inputs
            # Generate the passed down tensor and each layer tensor
            passed_down, every_layer = LLMFunc.pre_process(pre_inputs)
            # Store the passed down and each layer tensor
            self.micro_batch_passed_down[i] = passed_down
            self.micro_batch_every_layer[i] = every_layer
        self.every_layer_len = len(every_layer)

    @torch.no_grad()
    def _conduct_forward(self, task: LLMTask):
        config = get_flextrain_config()
        scheduler = self.scheduler

        # Need to fetch new unit parameters
        if scheduler.new_unit_entered:
            # Conduct pre-process
            if scheduler.in_first_unit:
                # Warm up the forward pipeline here for better overlapping
                self.model_coordinator.warmup_forward_pipeline()
                # Conduct the batch pre-process
                self._batch_pre_process()
            # End of in_first_unit

            # Fetch the new unit parameters
            self.model_coordinator.pre_forward_unit(task.unit)
        # End of new_unit_entered

        # Load recent passed down and each layer tensor
        passed_down = self.micro_batch_passed_down[task.micro_batch]
        every_layer = self.micro_batch_every_layer[task.micro_batch]

        # Conduct forward
        passed_down = detach_variable(passed_down)
        start = task.unit * config.checkpoint_interval
        end = min(start + config.checkpoint_interval, config.num_layers)
        passed_down, ctx = checkpointed_forward(
            LLMFunc.layer_forward(start, end),
            *passed_down, *every_layer
        )

        # Save recent passed down tensor and context
        self.micro_batch_passed_down[task.micro_batch] = passed_down
        self.micro_batch_unit_ctx[task.micro_batch][task.unit] = ctx

    @torch.enable_grad()
    def _conduct_last_unit(self, task: LLMTask):
        config = get_flextrain_config()

        # 1. Conduct forward of the last unit
        passed_down = self.micro_batch_passed_down[task.micro_batch]
        every_layer = self.micro_batch_every_layer[task.micro_batch]
        start = task.unit * config.checkpoint_interval
        end = min(start + config.checkpoint_interval, config.num_layers)
        passed_down = detach_variable(passed_down)
        last_passed_down = LLMFunc.layer_forward(start, end)(
            *passed_down, *every_layer
        )

        # 2. Conduct post-process
        llm_outputs = LLMFunc.post_process(
            last_passed_down,
            self.micro_batch_post_inputs[task.micro_batch]
        )

        # 3. Compute loss
        llm_loss_rsts = LLMFunc.loss(
            llm_outputs,
            self.micro_batch_loss_inputs[task.micro_batch]
        )
        self.micro_batch_loss_rsts[task.micro_batch] = llm_loss_rsts

        # 4. Conduct backward from loss to pre-post_process
        loss = retrieve_llm_loss(llm_loss_rsts)

        for w in range(dist.get_world_size()):
            dist.barrier()
            if dist.get_rank() == w:
                print(loss)
                if w == dist.get_world_size() - 1:
                    print()
            dist.barrier()

        torch.autograd.backward(loss)

        # 5. Get the gradients
        passed_back = retrieve_tensor_grads(passed_down)
        passed_back += [None] * self.every_layer_len
        self.micro_batch_passed_back[task.micro_batch] = passed_back

    def _conduct_backward(self, task: LLMTask):
        scheduler = self.scheduler

        # Need to fetch new unit parameters
        if scheduler.new_unit_entered:
            self.model_coordinator.pre_backward_unit(task.unit)
        # End of new_unit_entered

        # Last unit needs special treatment
        if scheduler.in_last_unit:
            return self._conduct_last_unit(task)
        # End of in_last_unit

        # Conduct backward
        ctx = self.micro_batch_unit_ctx[task.micro_batch][task.unit]
        passed_back = self.micro_batch_passed_back[task.micro_batch]
        passed_back = checkpointed_backward(ctx, *passed_back)
        self.micro_batch_passed_back[task.micro_batch] = passed_back

    def step(self):
        return


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


# for i in range(3):
        #     pre_input, post_input, loss_input = llm_funcs.get_batch()

        #     self.model_coordinator.warmup_forward_pipeline()
        #     passed_down, every_layer = llm_funcs.pre_process(pre_input)

        #     ctxs = []
        #     for j in range(24):
        #         passed_down = detach_variable(passed_down)
        #         self.model_coordinator.pre_forward_unit(j)
        #         passed_down, ctx = checkpointed_forward(
        #             llm_funcs.layer_forward(j, j + 1),
        #             passed_down, every_layer
        #         )
        #         self.model_coordinator.post_forward_unit(j)
        #         ctxs.append(ctx)

        #     passed_down = detach_variable(passed_down)
        #     llm_output = llm_funcs.post_process(passed_down, post_input)
        #     loss, loss_store = llm_funcs.loss(llm_output, loss_input)

        #     for w in range(dist.get_world_size()):
        #         dist.barrier()
        #         if dist.get_rank() == w:
        #             print(loss)
        #             if w == dist.get_world_size() - 1:
        #                 print()
        #         dist.barrier()
        # assert False


# passed_down = detach_variable(passed_down)
            # llm_output = llm_funcs.post_process(passed_down, post_input)
            # loss, loss_store = llm_funcs.loss(llm_output, loss_input)

            # # backward from loss to pre-post_process
            # torch.autograd.backward(loss)

            # # get the gradients
            # passed_back = retrieve_tensor_grads(passed_down)
            # passed_back += [None] * len(every_layer)

            # ctxs.reverse()
            # for ctx in ctxs:
            #     passed_back = checkpointed_backward(ctx, *passed_back)

            # # do the backward of the pre-process
            # torch.autograd.backward(pre_out[0], passed_back[0])
