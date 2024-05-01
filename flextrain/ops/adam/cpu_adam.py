# Parts of the code here are adapted from DeepSpeed
#
# Repository: https://github.com/microsoft/DeepSpeed
# File: deepspeed/ops/adam/cpu_adam.py
# Commit: 0d9cfa0
# License: Apache-2.0

import torch
from cpuinfo import get_cpu_info
from flextrain.utils import rank0_logger
from flextrain.ops.op_builder import CPUAdamBuilder


class FlexTrainCPUAdam(torch.optim.Optimizer):
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
        super(FlexTrainCPUAdam, self).__init__(model_params, default_args)

        cpu_info = get_cpu_info()
        self.cpu_vendor = cpu_info["vendor_id_raw"].lower() \
            if "vendor_id_raw" in cpu_info else "unknown"
        if "amd" in self.cpu_vendor:
            for group_id, group in enumerate(self.param_groups):
                for param_id, p in enumerate(group['params']):
                    if p.dtype == torch.half:
                        rank0_logger.warning(
                            "FP16 params for CPUAdam may not work on AMD CPUs"
                        )
                        break
                else:
                    continue
                break

        self.opt_id = FlexTrainCPUAdam.optimizer_id
        FlexTrainCPUAdam.optimizer_id += 1
        self.adam_w_mode = adamw_mode
        self.fp32_optimizer_states = fp32_optimizer_states
        self.ft_opt_adam = CPUAdamBuilder().load()

        self.ft_opt_adam.create_adam(
            self.opt_id, lr, betas[0], betas[1],
            eps, weight_decay, adamw_mode, True
        )

    def __del__(self):
        try:
            self.ft_opt_adam.destroy_adam(self.opt_id)
        except BaseException:
            pass

    def __setstate__(self, state):
        super(FlexTrainCPUAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, closure=None):
        """ Update the model parameters.

        Args:
            closure (callable, optional): closure to compute the loss.
                Defaults to ``None``.

        Returns:
            loss: if ``closure`` is provided. Otherwise ``None``.
        """

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # intended device for step
        device = torch.device('cpu')

        for group_id, group in enumerate(self.param_groups):
            for param_id, p in enumerate(group['params']):

                if p.grad is None:
                    continue

                assert p.device == device

                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # use full precision by default
                    # unless self.fp32_optimizer_states is off
                    state_dtype = torch.float \
                        if self.fp32_optimizer_states else p.dtype
                    # gradient momentums
                    state['exp_avg'] = torch.zeros_like(
                        p.data, dtype=state_dtype, device=device)
                    # gradient variances
                    state['exp_avg_sq'] = torch.zeros_like(
                        p.data, dtype=state_dtype, device=device)

                state['step'] += 1
                beta1, beta2 = group['betas']

                self.ft_opt_adam.adam_update(
                    self.opt_id, state['step'], group['lr'],
                    beta1, beta2, group['eps'],
                    group['weight_decay'], group['bias_correction'],
                    p.data, p.grad.data,
                    state['exp_avg'], state['exp_avg_sq']
                )

        return loss
