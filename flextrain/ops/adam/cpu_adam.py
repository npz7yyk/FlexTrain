# Parts of the code here are adapted from DeepSpeed
#
# Repository: https://github.com/microsoft/DeepSpeed
# File: deepspeed/ops/adam/cpu_adam.py
# Commit: 0d9cfa0
# License: Apache-2.0

import torch

from cpuinfo import get_cpu_info
from torch import Tensor
from typing import Dict

from flextrain.config import get_flextrain_config
from flextrain.optimizer import FlexTrainCPUOptimizer
from flextrain.ops.op_builder import CPUAdamBuilder
from flextrain.utils import rank0_logger


class FlexTrainCPUAdam(FlexTrainCPUOptimizer, torch.optim.Optimizer):
    optimizer_id = 0

    def __init__(
        self,
        model_params,
        lr=1e-3,
        bias_correction=True,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        amsgrad=False,
        adamw_mode=True,
        *args, **kwargs
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
        """

        self.default_args = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            bias_correction=bias_correction,
            amsgrad=amsgrad
        )
        # Initialize the optimizer
        FlexTrainCPUOptimizer.__init__(self)
        torch.optim.Optimizer.__init__(self, model_params, self.default_args)

        cpu_info = get_cpu_info()
        self.cpu_vendor = cpu_info["vendor_id_raw"].lower() \
            if "vendor_id_raw" in cpu_info else "unknown"
        master_dtype = get_flextrain_config().mixed_precision.master_dtype
        if "amd" in self.cpu_vendor and master_dtype == torch.float16:
            rank0_logger.warning("FP16 for CPUAdam may not work on AMD CPUs")

        self.opt_id = FlexTrainCPUAdam.optimizer_id
        FlexTrainCPUAdam.optimizer_id += 1
        self.adam_w_mode = adamw_mode
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

    def _set_defaults_args(self, group_args: Dict):
        for k, v in self.default_args.items():
            if k not in group_args:
                group_args[k] = v

    def _init_optimizer_states(self, numel: int, dtype: torch.dtype):
        # Create exp_avg and exp_avg_sq
        return (
            torch.zeros(numel, dtype=dtype),
            torch.zeros(numel, dtype=dtype)
        )

    def _step(
        self, group_args: Dict,
        parameter: Tensor, gradient: Tensor, *optimizer_states: Tensor
    ):
        # Ensure that the step key is present in the group arguments
        STEP_KEY = "step"
        assert STEP_KEY in group_args, \
            f"Key '{STEP_KEY}' not found in parameter group arguments."
        # Set default values for the group arguments if not present
        self._set_defaults_args(group_args)
        # Perform the Adam update
        self.cpu_adam.adam_update(
            self.opt_id, group_args["step"], group_args["lr"],
            *group_args["betas"], group_args["eps"],
            group_args["weight_decay"], group_args["bias_correction"],
            parameter, gradient, *optimizer_states
        )
