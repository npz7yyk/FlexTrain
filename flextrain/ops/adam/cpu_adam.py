# Parts of the code here are adapted from DeepSpeed
#
# Repository: https://github.com/microsoft/DeepSpeed
# File: deepspeed/ops/adam/cpu_adam.py
# Commit: 0d9cfa0
# License: Apache-2.0

import torch
from cpuinfo import get_cpu_info
from dataclasses import dataclass
from typing import List, Dict

from flextrain.config import get_flextrain_config
from flextrain.optimizer import (
    OptTarget,
    FlexTrainCPUOptimizer
)
from flextrain.ops.op_builder import CPUAdamBuilder
from flextrain.utils import rank0_logger


@dataclass
class AdamOptTarget(OptTarget):
    exp_avg: torch.Tensor
    exp_avg_sq: torch.Tensor


class FlexTrainCPUAdam(torch.optim.Optimizer, FlexTrainCPUOptimizer):
    optimizer_id = 0

    def __init__(
        self,
        unit_group_map: Dict[int, Dict],
        non_layerwise_params: List[torch.Tensor],
        non_layerwise_param_groups: List[Dict],
        model_params,
        lr=1e-3,
        bias_correction=True,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        amsgrad=False,
        adamw_mode=True,
        fp32_optimizer_states=True
    ):
        """
        Vectorized implementation of two variations of Adam optimizer on CPU.

        * Adam: A Method for Stochastic Optimization \
            (https://arxiv.org/abs/1412.6980)
        * AdamW: Fixing Weight Decay Regularization in Adam \
            (https://arxiv.org/abs/1711.05101)

        FlexTrain CPU Adam is adapted from DeepSpeed CPU Adam.
        DeepSpeed CPU Adam(W) provides between 5x to 7x speedup over CPU
        torch.optim.adam(W). In order to apply this optimizer, the model
        requires to have its master parameter reside on the CPU memory.

        For calling step function, there are two options available:
        (1) update optimizer's states
        (2) update optimizer's states and \
            copy the parameters back to GPU at the same time.
        We have seen that the second option can bring 30% higher throughput
        than the doing the copy separately using option one.

        Arguments:
            model_params (iterable): \
                iterable of parameters to optimize or \
                dicts defining parameter groups.
            lr (float, optional): learning rate. (default: 1e-3)
            betas (Tuple[float, float], optional): \
                coefficients used for computing running averages of \
                gradient and its square. (default: (0.9, 0.999))
            eps (float, optional): \
                term added to the denominator to improve \
                numerical stability. (default: 1e-8)
            weight_decay (float, optional): \
                weight decay (L2 penalty) (default: 0.01)
            amsgrad (boolean, optional): \
                whether to use the AMSGrad variant of this algorithm \
                from the paper `On the Convergence of Adam and Beyond` \
                (default: False) NOT SUPPORTED in DeepSpeed CPUAdam!
            adamw_mode: \
                select between Adam and AdamW implementations (default: AdamW)
            fp32_optimizer_states: \
                creates momentum and variance in full precision \
                regardless of the precision of the parameters (default: True)
        """

        default_args = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            bias_correction=bias_correction,
            amsgrad=amsgrad
        )
        # Initialize the optimizer
        # model_params, default_args for torch.optim.Optimizer
        # unit_group_map for FlexTrainCPUOptimizer
        torch.optim.Optimizer.__init__(self, model_params, default_args)
        FlexTrainCPUOptimizer.__init__(
            self,
            unit_group_map=unit_group_map,
            non_layerwise_params=non_layerwise_params,
            non_layerwise_param_groups=non_layerwise_param_groups
        )

        cpu_info = get_cpu_info()
        self.cpu_vendor = cpu_info["vendor_id_raw"].lower() \
            if "vendor_id_raw" in cpu_info else "unknown"
        master_dtype = get_flextrain_config().mixed_precision.master_dtype
        if "amd" in self.cpu_vendor and master_dtype == torch.float16:
            rank0_logger.warning("FP16 for CPUAdam may not work on AMD CPUs")

        self.opt_id = FlexTrainCPUAdam.optimizer_id
        FlexTrainCPUAdam.optimizer_id += 1
        self.adam_w_mode = adamw_mode
        self.fp32_optimizer_states = fp32_optimizer_states
        self.cpu_adam = CPUAdamBuilder().load()

        self.cpu_adam.create_adam(
            self.opt_id, lr, betas[0], betas[1],
            eps, weight_decay, adamw_mode, True
        )

    def __del__(self):
        try:
            self.cpu_adam.destroy_adam(self.opt_id)
        except BaseException:
            pass

    def __setstate__(self, state):
        super(FlexTrainCPUAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def _step(self, step: int, args: Dict, opt_target: AdamOptTarget):
        beta1, beta2 = args['betas']
        self.cpu_adam.adam_update(
            self.opt_id, step, args['lr'],
            beta1, beta2, args['eps'],
            args['weight_decay'], args['bias_correction'],
            opt_target.para.data, opt_target.grad.data,
            opt_target.exp_avg.data, opt_target.exp_avg_sq.data
        )

    @torch.no_grad()
    def profile_step(self, numel: int, dtype: torch.dtype):
        if not hasattr(self, 'curr_step'):
            self.curr_step = 0
            self.para = torch.empty(numel, dtype=dtype, pin_memory=True)
            self.grad = torch.empty(numel, dtype=dtype, pin_memory=True)
            self.exp_avg = torch.zeros_like(self.para, pin_memory=True)
            self.exp_avg_sq = torch.zeros_like(self.para, pin_memory=True)
        self.curr_step += 1
        self.cpu_adam.adam_update(
            self.opt_id, self.curr_step, 1e-3, 0.9, 0.999, 1e-8, 0.01, True,
            self.para.data, self.grad.data,
            self.exp_avg.data, self.exp_avg_sq.data
        )

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for p, g in zip(
            self._non_layerwise_params, self._non_layerwise_param_groups
        ):
            if p.grad is None:
                continue
            if p.numel() == 0:
                continue
            if p.grad.data.is_sparse:
                raise RuntimeError(
                    'CPUAdam does not support sparse gradients, '
                    'please consider SparseAdam instead'
                )

            state = self.state[p]
            # State initialization
            if len(state) == 0:
                assert 'step' in g, "step should be in the group"
                # Exponential moving average of gradient values
                state['exp_avg'] = torch.zeros_like(p.data)
                # Exponential moving average of squared gradient values
                state['exp_avg_sq'] = torch.zeros_like(p.data)

            self._step(g['step'], g, AdamOptTarget(
                p.data, p.grad.data,
                state['exp_avg'].data, state['exp_avg_sq'].data
            ))

        return loss

    def submit_step(
        self, unit_index: int, para: torch.Tensor, grad: torch.Tensor,
        exp_avg: torch.Tensor, exp_avg_sq: torch.Tensor
    ):
        return self._submit_step(
            unit_index, AdamOptTarget(para, grad, exp_avg, exp_avg_sq)
        )
