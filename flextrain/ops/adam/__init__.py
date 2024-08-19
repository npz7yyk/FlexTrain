from typing import List, Dict

from flextrain.optimizer import FlexTrainOptimizer

from .cpu_adam import FlexTrainCPUAdam
from .fused_adam import FusedAdam as FlexTrainGPUAdam


class FlexTrainAdam(FlexTrainOptimizer):
    # The number of optimizer states for each parameter.
    EACH_NUMEL_NUM_STATES = 2

    def __init__(
        self,
        param_groups: List[Dict],
        lr=1e-3,
        bias_correction=True,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        amsgrad=False,
        adamw_mode=True,
        fp32_optimizer_states=True,
        set_grad_none=True
    ):
        # 1. Group the parameters and their arguments.
        #    Initialize the optimizer coordinator.
        super().__init__(
            param_groups=param_groups,
            each_numel_num_states=FlexTrainAdam.EACH_NUMEL_NUM_STATES
        )

        # 2. Initialize GPU Adam optimizer.
        self.gpu_optimizer = FlexTrainGPUAdam(
            self.param_groups,
            lr=lr,
            bias_correction=bias_correction,
            betas=betas,
            eps=eps,
            adam_w_mode=adamw_mode,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            set_grad_none=set_grad_none
        )
        self.gpu_optimizer.link_layerwise_parameters(self.gpu_layerwise_states)

        # 3. Initialize CPU Adam optimizer.
        self.cpu_optimizer = FlexTrainCPUAdam(
            self.unit_group_map,
            self.param_groups,
            lr=lr,
            bias_correction=bias_correction,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            adamw_mode=adamw_mode,
            fp32_optimizer_states=fp32_optimizer_states
        ) if self.is_cpu_optimizer_needed() else None
