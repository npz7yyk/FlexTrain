from typing import List, Dict

from flextrain.optimizer import FlexTrainOptimizer

from .cpu_adam import FlexTrainCPUAdam


class FlexTrainAdam(FlexTrainOptimizer):
    # The number of optimizer states for each parameter.
    opt_state_per_element = 2

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
            opt_state_per_element=FlexTrainAdam.opt_state_per_element
        )

        # 2. Initialize CPU Adam optimizer.
        self.cpu_optimizer = FlexTrainCPUAdam(
            self.param_groups,
            lr=lr,
            bias_correction=bias_correction,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            adamw_mode=adamw_mode,
            fp32_optimizer_states=fp32_optimizer_states
        )
