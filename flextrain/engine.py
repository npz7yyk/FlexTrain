import torch

from torch.nn import Module
from typing import Any, Callable, Sequence, Tuple, List, Dict

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
from flextrain.loss_scaler import create_loss_scaler
from flextrain.memory.coordinator import (
    get_data_stream,
    get_para_coordinator,
    get_opt_coordinator,
    get_interlayer_coordinator,
    InterLayerTask,
    retrieve_tensor
)
from flextrain.optimizer import FlexTrainOptimizer
from flextrain.scheduler import LLMTask, GreedySnakeBlockScheduler


class LRScheduler:
    def step(self, **kwargs):
        pass


class FlexTrainEngine(object):
    """
    FlexTrainEngine is designed to maximize the training throughput.
    ZeRO Stage 3 optimization is used to support multi-GPU training.
    """

    def __init__(
        self,
        model: Module,
        optimizer: FlexTrainOptimizer,
        lr_scheduler: LRScheduler = None
    ):
        super().__init__()

        # Get FlexTrain configuration
        config = get_flextrain_config()

        # Link to model
        # Logically move the model to GPU and set the dtype
        self.model = model.cuda().to(dtype=config.mixed_precision.device_dtype)

        # Link to optimizer
        assert isinstance(optimizer, FlexTrainOptimizer), (
            "FlexTrainOptimizer is required to initialize FlexTrainEngine. "
            "Try to explicitly warp the optimizer with FlexTrainOptimizer."
        )
        self.optimizer = optimizer

        # Link to parameter, interlayer and optimizer coordinator.
        self.data_stream = get_data_stream()
        self.para_coordinator = get_para_coordinator()
        self.interlayer_coordinator = get_interlayer_coordinator()
        self.opt_coordinator = get_opt_coordinator()
        self.opt_coordinator.initialize(
            self.optimizer.cpu_optimizer,
            self.optimizer.opt_state_per_element
        )

        # LLM training information
        self.micro_batch_per_batch = self._compute_micro_batch_per_batch()
        self.num_units = self.para_coordinator.num_units

        # Create computation scheduler
        self.scheduler = GreedySnakeBlockScheduler(
            self.micro_batch_per_batch,
            self.num_units
        )

        # Link / Create loss scaler
        self.custom_loss_scaler = False
        self.loss_scaler = create_loss_scaler()

        # Link to lr_scheduler
        self.lr_scheduler = lr_scheduler

        # Post-forward / pre-backward functions
        self.delayed_functions: List[Callable] = []

    def _compute_micro_batch_per_batch(self):
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
            for _ in range(self.micro_batch_per_batch)
        ]

        # Loss result container
        self.micro_batch_loss_rsts: Dict[int, Sequence[Any]] = {}

        # Last checkpoint / gradient to offload
        self.ckpt_offload: InterLayerTask = None
        self.grad_offload: InterLayerTask = None

    @torch.no_grad()
    def _conduct_first_unit(self, task: LLMTask):
        config = get_flextrain_config()

        # Unpack a micro batch
        pre_inputs, post_inputs, loss_inputs = LLMFunc.get_batch()
        # Store the inputs
        self.micro_batch_post_inputs[task.micro_batch] = post_inputs
        self.micro_batch_loss_inputs[task.micro_batch] = loss_inputs

        def forward_func():
            # Generate the passed down tensor and each layer tensor
            passed_down, every_layer = LLMFunc.pre_process(pre_inputs)

            # Store the passed down and each layer tensor
            self.micro_batch_passed_down[task.micro_batch] = passed_down
            self.micro_batch_every_layer[task.micro_batch] = every_layer
            self.every_layer_len = len(every_layer)

            # Conduct forward
            start = task.unit * config.checkpoint_interval
            end = start + config.checkpoint_interval
            return LLMFunc.layer_forward(start, end)(
                *passed_down, *every_layer
            )

        # Conduct checkpointed forward and return passed_down & ctx
        return checkpointed_forward(forward_func)

    @torch.no_grad()
    def _conduct_forward(self, task: LLMTask):
        config = get_flextrain_config()
        scheduler = self.scheduler

        # Submit prefetching and offloading tasks
        # 1. Prefetch the next passed_down and offload the last passed_down
        next_task = self.scheduler.next_task
        # If the next task is in the same micro batch, prefetch is not needed
        ckpt_prefetch = None if next_task.micro_batch == task.micro_batch \
            else InterLayerTask(next_task.unit - 1, next_task.micro_batch)
        self.interlayer_coordinator.pre_micro_batch_forward(
            ckpt_prefetch, self.ckpt_offload
        )
        # 2. Prefetch the parameter of the next unit
        self.para_coordinator.pre_micro_batch_forward(
            task.unit, task.micro_batch
        )
        # Link parameters to memory (prefetch NVMe parameters if needed)
        if scheduler.new_unit_entered:
            self.para_coordinator.pre_unit_forward(task.unit)
        # End of submit prefetching and offloading tasks

        # Wait for all in-flight operations
        self.data_stream.synchronize()
        # Execute just submitted operations
        self.data_stream.execute()

        # Conduct pre-process if in the first unit
        if scheduler.in_first_unit:
            passed_down, ctx = self._conduct_first_unit(task)
        else:
            # Load recent passed down and each layer tensor
            passed_down = self.micro_batch_passed_down[task.micro_batch]
            every_layer = self.micro_batch_every_layer[task.micro_batch]
            passed_down_tensor = retrieve_tensor(passed_down)
            # If passed_down is empty, use the available layer checkpoint
            if passed_down_tensor.numel() == 0:
                passed_down_tensor.data = \
                    self.interlayer_coordinator.available_layer_ckpt

            # Conduct forward
            passed_down = detach_variable(passed_down)
            start = task.unit * config.checkpoint_interval
            end = start + config.checkpoint_interval
            passed_down, ctx = checkpointed_forward(
                LLMFunc.layer_forward(start, end),
                *passed_down, *every_layer
            )

        # Save recent passed down tensor and context
        self.micro_batch_passed_down[task.micro_batch] = passed_down
        self.micro_batch_unit_ctx[task.micro_batch][task.unit] = ctx
        # Record the checkpoint to offload
        self.ckpt_offload = InterLayerTask(
            task.unit, task.micro_batch, retrieve_tensor(passed_down)
        )

    @torch.enable_grad()
    def _conduct_last_unit(self, task: LLMTask):
        config = get_flextrain_config()

        # 0. Conduct post-forward / pre-backward functions
        self._dalayed_step()

        # 1. Conduct forward of the last unit
        passed_down = self.micro_batch_passed_down[task.micro_batch]
        every_layer = self.micro_batch_every_layer[task.micro_batch]
        passed_down_tensor = retrieve_tensor(passed_down)
        # If passed_down is empty, use the available layer checkpoint
        if passed_down_tensor.numel() == 0:
            passed_down_tensor.data = \
                self.interlayer_coordinator.available_layer_ckpt

        start = task.unit * config.checkpoint_interval
        end = min(
            start + config.checkpoint_interval,
            self.para_coordinator.num_layers
        )
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
        llm_loss_rst = LLMFunc.loss(
            llm_outputs,
            self.micro_batch_loss_inputs[task.micro_batch]
        )
        self.micro_batch_loss_rsts[task.micro_batch] = llm_loss_rst

        dist.print_rank_by_rank(f"Loss: {llm_loss_rst}")

        # 4. Scale the loss
        loss = retrieve_llm_loss(llm_loss_rst)
        loss = loss.float() * self.loss_scale / self.micro_batch_per_batch

        # 5. Conduct backward from loss to pre-post_process
        torch.autograd.backward(loss)

        # 6. Get the gradients
        passed_back = retrieve_tensor_grads(passed_down)
        self.micro_batch_passed_back[task.micro_batch] = passed_back

        return passed_back

    def _conduct_backward(self, task: LLMTask):
        scheduler = self.scheduler

        # Submit prefetching the next passed_down
        next_task = self.scheduler.next_task
        ckpt_prefetch = InterLayerTask(
            next_task.unit - 1, next_task.micro_batch
        )
        grad_prefetch = None if next_task.micro_batch == task.micro_batch \
            else InterLayerTask(next_task.unit, next_task.micro_batch)
        self.interlayer_coordinator.pre_micro_batch_backward(
            ckpt_prefetch, grad_prefetch, self.grad_offload
        )
        # End of submit prefetching the next passed_down

        # 2. Prefetch the parameter of the next unit
        self.para_coordinator.pre_micro_batch_backward(
            task.unit, task.micro_batch
        )
        self.opt_coordinator.pre_micro_batch_backward(
            task.unit, task.micro_batch
        )
        # Link parameters to memory (prefetch NVMe parameters if needed)
        if scheduler.new_unit_entered:
            self.para_coordinator.pre_unit_backward(task.unit)
            self.opt_coordinator.pre_unit_backward(task.unit)
        # End of submit prefetching and offloading tasks

        # Wait for all in-flight operations
        self.data_stream.synchronize()
        # Execute just submitted operations
        self.data_stream.execute()

        # Last unit needs special treatment
        if scheduler.in_last_unit:
            passed_back = self._conduct_last_unit(task)
        else:
            # Link to the checkpoint buffer if not in the first unit
            ctx = self.micro_batch_unit_ctx[task.micro_batch][task.unit]
            if not scheduler.in_first_unit:
                # fwd_args[0] is passed_down, except for the first unit
                checkpoint_tensor = retrieve_tensor(ctx.fwd_args[0])
                checkpoint_tensor.data = \
                    self.interlayer_coordinator.available_layer_ckpt

            passed_back = self.micro_batch_passed_back[task.micro_batch]
            passed_back_tensor = retrieve_tensor(passed_back)
            # If passed_back is empty, use the available layer gradient
            if passed_back_tensor.numel() == 0:
                passed_back_tensor.data = \
                    self.interlayer_coordinator.available_layer_grad
            passed_back = checkpointed_backward(ctx, *passed_back)

            # Exclude every layer tensor in the passed back
            passed_back = passed_back[:-self.every_layer_len]
            self.micro_batch_passed_back[task.micro_batch] = passed_back
        # End of conduct backward

        # Record the gradient to offload
        # First unit does not have tensor input, thus no gradient to offload
        self.grad_offload = InterLayerTask(
            task.unit - 1, task.micro_batch, retrieve_tensor(passed_back)
        ) if not scheduler.in_first_unit else None

        # Conduct gradient accumulation
        self.opt_coordinator.post_micro_batch_backward(
            task.unit, task.micro_batch
        )

    def override_loss_scale(self, loss_scale: float):
        self.custom_loss_scaler = True
        self.external_loss_scale = loss_scale

    def _get_loss_scale(self):
        if self.custom_loss_scaler:
            return self.external_loss_scale
        else:
            return self.loss_scaler.cur_scale

    def _set_loss_scale(self, value):
        self.loss_scaler.cur_scale = value

    loss_scale = property(_get_loss_scale, _set_loss_scale)

    @property
    def dynamic_loss_scaling(self):
        return self.loss_scaler.dynamic

    def _warmup_forward_pipeline(self):
        self.para_coordinator.warmup_forward_pipeline()

    def step(
        self, *,  # Enforce keyword-only arguments
        lr_kwargs: Dict = None
    ):
        # Conduct on the spot tasks
        pass
        # End of on the spot tasks

        # Submit delayed tasks
        def _update_lr():
            if self.lr_scheduler is not None and lr_kwargs is not None:
                self.lr_scheduler.step(**lr_kwargs)

        self.delayed_functions.append(_update_lr)
        self.delayed_functions.append(self.optimizer.update_state)
        # End of submit delayed tasks

        return

    def _dalayed_step(self):
        # Conduct delayed tasks
        for func in self.delayed_functions:
            func()

        # Clear delayed tasks
        self.delayed_functions.clear()

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def train_iteration(self):
        # 1. Conduct reset
        assert self.data_stream.is_empty()
        self._reset_context()

        import time
        times_fwd = []
        times_bwd = []

        # 2. Conduct all tasks assigned by the scheduler
        # self.para_coordinator.warmup_forward_pipeline()
        self._warmup_forward_pipeline()
        for task in self.scheduler:
            if task.is_forwarding:
                torch.cuda.nvtx.range_push(f"Forward unit {task.unit}")
                t_start = time.time()
                self._conduct_forward(task)
                t_end = time.time()
                times_fwd.append(t_end - t_start)
                torch.cuda.nvtx.range_pop()
            else:
                t_start = time.time()
                torch.cuda.nvtx.range_push(f"Backward unit {task.unit}")
                self._conduct_backward(task)
                t_end = time.time()
                times_bwd.append(t_end - t_start)
                torch.cuda.nvtx.range_pop()
        self.para_coordinator.clear_backward_pipeline()
        self.opt_coordinator.clear_backward_pipeline()

        times = sorted(times_fwd, reverse=True)
        times = times[5:]
        avg_time = sum(times) / len(times)
        dist.print_rank_by_rank(avg_time * 1000)

        times = sorted(times_bwd, reverse=True)
        times = times[5:]
        avg_time = sum(times) / len(times)
        dist.print_rank_by_rank(avg_time * 1000)

        for p in self.optimizer.non_layerwise_params:
            dist.print_rank0(p.grad.flatten()[:10])

        # 3. Conduct optimizer step
        self.optimizer.step()

        # 4. Return the loss results
        loss_rsts = []
        for i in range(self.micro_batch_per_batch):
            loss_rsts.append(self.micro_batch_loss_rsts[i])

        if not hasattr(self, 'count'):
            self.count = 0
        self.count += 1
        if self.count % 10 == 0:
            exit()

        return loss_rsts
