"""
Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License").
Taken and modified for FlexTrain from:
https://github.com/NVIDIA/Megatron-LM/blob/master/fp16/loss_scaler.py
Commit: 93ab4bea59dc5cbf97c079d313741866af4deac9
"""

import torch
from abc import ABC

from flextrain.config import get_flextrain_config
from flextrain.utils import rank0_logger

INITIAL_LOSS_SCALE = 'init_scale'
SCALE_WINDOW = 'scale_window'
DELAYED_SHIFT = 'delayed_shift'
CONSECUTIVE_HYSTERESIS = 'consecutive_hysteresis'
MIN_LOSS_SCALE = 'min_scale'


# item() is a recent addition, so this helps with backward compatibility.
def to_python_float(tensor: torch.Tensor) -> float:
    if hasattr(tensor, 'item'):
        return tensor.item()
    assert tensor.nelement() == 1
    return tensor[0]


class LossScalerBase(ABC):
    """ Base class for loss scalers. """

    def __init__(self, cur_scale: float):
        self.cur_scale = cur_scale
        self.dynamic = False

    @property
    def loss_scale(self):
        return self.cur_scale

    def scale_gradient(self, module, grad_in, grad_out):
        return tuple(self.loss_scale * g for g in grad_in)

    def update_scale(self, overflow):
        pass

    def backward(self, loss: torch.Tensor, retain_graph=False):
        scaled_loss = loss * self.loss_scale
        scaled_loss.backward(retain_graph=retain_graph)


class LossScaler(LossScalerBase):
    """ Class that manages a static loss scale

    Args:
        scale (float, optional, default=1.0): The loss scale.
    """

    def __init__(self, scale=1):
        super(LossScaler, self).__init__(scale)

    def has_overflow(self, params):
        return False

    def _has_inf_or_nan(x):
        return False


class DynamicLossScaler(LossScalerBase):
    """
    Class that manages dynamic loss scaling. \
    It is recommended to use :class:`DynamicLossScaler` indirectly, \
    by supplying ``dynamic_loss_scale=True`` to the constructor of \
    :class:`FlexTrainEngine`.
    However, it's important to understand how :class:`DynamicLossScaler` \
    operates, because the default options can be changed using the \
    ``dynamic_loss_args`` argument to :class:`FlexTrainEngine`'s constructor.

    Loss scaling is designed to combat the problem of underflowing gradients \
    encountered at long times when training fp16 networks. \
    Dynamic loss scaling begins by attempting a very high loss scale. \
    Ironically, this may result in OVERflowing gradients. If overflowing \
    gradients are encountered, :class:`DynamicLossScaler` informs \
    :class:`FlexTrainEngine` that an overflow has occurred. \
    :class:`FlexTrainEngine` then skips the update step for this particular \
    iteration/minibatch, and :class:`DynamicLossScaler` adjusts the loss \
    scale to a lower value. If a certain number of iterations occur without \
    overflowing gradients detected, :class:`DynamicLossScaler` increases the \
    loss scale once more. In this way :class:`DynamicLossScaler` attempts to \
    "ride the edge" of always using the highest loss scale possible without \
    incurring overflow.

    Args:
        init_scale (float):
            Initial loss scale attempted by :class:`DynamicLossScaler.`
        scale_factor (float):
            Factor used when adjusting the loss scale. \
            If an overflow is encountered, the loss scale is readjusted to \
            loss scale/``scale_factor``. If ``scale_window`` consecutive \
            iterations take place without an overflow, the loss scale is \
            readjusted to loss_scale*``scale_factor``.
        scale_window (int):
            Number of consecutive iterations without an overflow to wait \
            before increasing the loss scale.
        consecutive_hysteresis (bool):
            Whether to refill hysteresis \
            if we reach an iteration that doesn't overflow
    """

    def __init__(
        self,
        init_scale,
        scale_factor=2.0,
        scale_window=1000,
        min_scale=1.0,
        delayed_shift=1,
        consecutive_hysteresis=False,
        raise_error_at_min_scale=False,
        dtype=torch.half
    ):
        super(DynamicLossScaler, self).__init__(init_scale)
        self.cur_iter = 0
        self.last_overflow_iter = -1
        self.scale_factor = scale_factor
        self.scale_window = scale_window
        self.min_scale = min_scale
        self.delayed_shift = delayed_shift
        self.cur_hysteresis = delayed_shift
        self.consecutive_hysteresis = consecutive_hysteresis
        self.raise_error_at_min_scale = raise_error_at_min_scale
        self.dynamic = True
        self.dtype = dtype

    def update_scale(self, overflow):
        if overflow:
            if self.delayed_shift == 1 or self.cur_hysteresis == 1:
                if (self.cur_scale == self.min_scale) \
                        and self.raise_error_at_min_scale:
                    raise Exception(
                        "Current loss scale already at minimum - "
                        "cannot decrease scale anymore. Exiting run."
                    )
                else:
                    next_scale = max(
                        self.cur_scale / self.scale_factor, self.min_scale
                    )
                    overflow_msg = "[flextrain] OVERFLOW! Skipping step."
                    if self.dtype == torch.half:
                        overflow_msg += (
                            f" Attempted loss scale: {int(self.cur_scale)},"
                            f" reducing to {int(next_scale)}."
                        )
                    rank0_logger.info(overflow_msg)
                    self.cur_scale = next_scale
            else:
                overflow_msg = "[flextrain] OVERFLOW! Skipping step."
                if self.dtype == torch.half:
                    overflow_msg += (
                        f" Attempted loss scale: {int(self.cur_scale)},"
                        f" but hysteresis is {self.cur_hysteresis}."
                        f" Reducing hysteresis to {self.cur_hysteresis - 1}."
                    )
                rank0_logger.info(overflow_msg)
                self.cur_hysteresis -= 1
            self.last_overflow_iter = self.cur_iter
        else:
            if self.consecutive_hysteresis:
                hysteresis_msg = (
                    f"Consecutive hysteresis is enabled. "
                    f"Restoring hysteresis to {self.delayed_shift}."
                )
                rank0_logger.info(hysteresis_msg)
                self.cur_hysteresis = self.delayed_shift
            delta_iter = self.cur_iter - self.last_overflow_iter
            if delta_iter % self.scale_window == 0:
                if not self.consecutive_hysteresis:
                    self.cur_hysteresis = self.delayed_shift
                self.cur_scale *= self.scale_factor
        self.cur_iter += 1


# Although loss scaling is defined for fp16, yet for backwards compatibility
# we still create a scaler for other dtypes which does not perform any scaling.
def create_loss_scaler():
    loss_scale_config = get_flextrain_config().mixed_precision
    init_loss_scale = 2 ** loss_scale_config.initial_scale_power
    dtype = loss_scale_config.device_dtype
    static_loss_scale = init_loss_scale,
    dynamic_loss_scale_enabled = loss_scale_config.dynamic_loss_scaling,
    dynamic_loss_args = {
        INITIAL_LOSS_SCALE: init_loss_scale,
        SCALE_WINDOW: loss_scale_config.loss_scale_window,
        MIN_LOSS_SCALE: loss_scale_config.min_loss_scale,
        DELAYED_SHIFT: loss_scale_config.hysteresis,
        CONSECUTIVE_HYSTERESIS:
            loss_scale_config.consecutive_hysteresis
    }
    if dtype == torch.half and dynamic_loss_scale_enabled:
        if dynamic_loss_args is None:
            return DynamicLossScaler(dtype=dtype)
        return DynamicLossScaler(dtype=dtype, **dynamic_loss_args)

    loss_scale_value = static_loss_scale if dtype == torch.half else 1.0
    return LossScaler(scale=loss_scale_value)
