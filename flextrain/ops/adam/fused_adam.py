# Parts of the code here are adapted from DeepSpeed
#
# Repository: https://github.com/microsoft/DeepSpeed
# File: deepspeed/ops/adam/fused_adam.py
# Commit: 0d9cfa0
# License: Apache-2.0

import torch
from torch import Tensor
from typing import Tuple, List

from flextrain.ops.op_builder import FusedAdamBuilder


class MultiTensorApply(object):

    def __init__(self, chunk_size):
        self.chunk_size = chunk_size

    def __call__(self, op, noop_flag_buffer, tensor_lists, *args):
        return op(self.chunk_size, noop_flag_buffer, tensor_lists, *args)


multi_tensor_applier = MultiTensorApply(2048 * 32)


class FusedAdam(torch.optim.Optimizer):
    """
    Implements Adam algorithm. Currently GPU-only.

    This version of fused Adam implements 2 fusions.

      * Fusion of the Adam update's elementwise operations
      * A multi-tensor apply launch that batches the elementwise updates \
        applied to all parameters into one or a few kernel launches.

    Adam was been proposed in `Adam: A Method for Stochastic Optimization`.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of grad and its square. (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability. (default: 1e-8)
        weight_decay (float, optional): L2 penalty (default: 0.01)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`
            (default: False) NOT SUPPORTED in FusedAdam!
        adam_w_mode (bool, optional): Apply L2 regularization or weight decay
            True for decoupled weight decay (AdamW) (default: True)
        set_grad_none (bool, optional): whether set grad to None when
            zero_grad() method is called. (default: True)
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        bias_correction=True,
        betas=(0.9, 0.999),
        eps=1e-8,
        adam_w_mode=True,
        weight_decay=0.01,
        amsgrad=False,
        set_grad_none=True
    ):

        if amsgrad:
            raise RuntimeError('AMSGrad variant not supported by FusedAdam')

        defaults = dict(
            lr=lr,
            bias_correction=bias_correction,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay
        )
        super(FusedAdam, self).__init__(params, defaults)
        self.adam_w_mode = 1 if adam_w_mode else 0
        self.set_grad_none = set_grad_none

        fused_adam_cuda = FusedAdamBuilder().load()
        # Skip buffer
        self._dummy_overflow_buf = torch.tensor(
            [0], dtype=torch.int, device=torch.cuda.current_device()
        )
        self.multi_tensor_adam = fused_adam_cuda.multi_tensor_adam

    def link_layerwise_parameters(self, param_states: List[Tuple[Tensor]]):
        # Link optimizer buffers to related parameters
        for p, m, v in param_states:
            state = self.state[p]
            state['step'] = 0
            state['exp_avg'] = m
            state['exp_avg_sq'] = v

    def zero_grad(self):
        if self.set_grad_none:
            for group in self.param_groups:
                for p in group['params']:
                    p.grad = None
        else:
            super(FusedAdam, self).zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            if len(group['params']) == 0:
                continue
            bias_correction = 1 if group['bias_correction'] else 0
            beta1, beta2 = group['betas']

            # assume same step across group now to simplify things
            # per parameter step can be easily support by making it tensor
            # or pass list into kernel
            if 'step' not in group:
                group['step'] = 0

            # create lists for multi-tensor apply
            g_16, p_16, m_16, v_16 = [], [], [], []
            g_bf, p_bf, m_bf, v_bf = [], [], [], []
            g_32, p_32, m_32, v_32 = [], [], [], []

            for p in group['params']:
                if p.grad is None:
                    continue
                if p.grad.data.is_sparse:
                    raise RuntimeError(
                        'FusedAdam does not support sparse gradients, '
                        'please consider SparseAdam instead'
                    )

                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state['step'] = group.get('step', 0)
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                if p.dtype == torch.float16:
                    g_16.append(p.grad.data)
                    p_16.append(p.data)
                    m_16.append(state['exp_avg'])
                    v_16.append(state['exp_avg_sq'])
                elif p.dtype == torch.bfloat16:
                    g_bf.append(p.grad)
                    p_bf.append(p)
                    m_bf.append(state['exp_avg'])
                    v_bf.append(state['exp_avg_sq'])
                elif p.dtype == torch.float32:
                    g_32.append(p.grad.data)
                    p_32.append(p.data)
                    m_32.append(state['exp_avg'])
                    v_32.append(state['exp_avg_sq'])
                else:
                    raise RuntimeError(
                        f'FusedAdam only support fp16, bf16 and fp32, '
                        f'get paramerter dtype: {p.dtype}'
                    )

            if len(g_16) > 0:
                state['step'] += 1
                multi_tensor_applier(
                    self.multi_tensor_adam, self._dummy_overflow_buf,
                    [g_16, p_16, m_16, v_16],
                    group['lr'], beta1, beta2, group['eps'], state['step'],
                    self.adam_w_mode, bias_correction, group['weight_decay']
                )

            if len(g_bf) > 0:
                state['step'] += 1
                multi_tensor_applier(
                    self.multi_tensor_adam, self._dummy_overflow_buf,
                    [g_bf, p_bf, m_bf, v_bf],
                    group['lr'], beta1, beta2, group['eps'], state['step'],
                    self.adam_w_mode, bias_correction, group['weight_decay']
                )

            if len(g_32) > 0:
                state['step'] += 1
                multi_tensor_applier(
                    self.multi_tensor_adam, self._dummy_overflow_buf,
                    [g_32, p_32, m_32, v_32],
                    group['lr'], beta1, beta2, group['eps'], state['step'],
                    self.adam_w_mode, bias_correction, group['weight_decay']
                )

        return loss
