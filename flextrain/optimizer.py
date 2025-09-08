import torch

from abc import ABC, abstractmethod
from collections.abc import Iterable
from torch import Tensor
from typing import List, Dict

from flextrain.config import get_flextrain_config
from flextrain.memory import move_into_contiguous
from flextrain.memory.coordinator import get_para_coordinator
from flextrain.param_group import (
    ParamGroup,
    GroupSegment,
    StepContext,
    create_group_segments,
    create_step_contexts
)
from flextrain.utils import dist

STEP_KEY = "step"
PARAMS_KEY = "params"


class FlexTrainCPUOptimizer(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def _init_optimizer_states(
        self, numel: int, dtype: torch.dtype
    ) -> List[Tensor]:
        pass

    @abstractmethod
    def _step(
        self, param_group: ParamGroup,
        half_parameter: Tensor | None, full_parameter: Tensor,
        gradient: Tensor, *optimizer_states: Tensor
    ):
        pass

    @torch.no_grad()
    def step(self, step_contexts: StepContext | List[StepContext]):
        """ Performs a single optimization step.

        Arguments:
            step_contexts (StepContext | List[StepContext]): \
                A context object that contains the target parameters, \
                gradients, and optimizer states, or a list of such objects. \
        """

        # Wrap the step contexts in a list if not already a list.
        if isinstance(step_contexts, StepContext):
            step_contexts = [step_contexts]

        for step_context in step_contexts:
            # Skip empty segments.
            if not step_context.gradient.numel():
                continue
            # Conduct the optimization step for each segment.
            self._step(
                step_context.param_group,
                step_context.half_parameter,
                step_context.full_parameter,
                step_context.gradient,
                *step_context.optimizer_states
            )

    @torch.no_grad()
    def profile_step(
        self, numel: int, device_dtype: torch.dtype, master_dtype: torch.dtype
    ):
        # Initialize the optimizer states.
        if not hasattr(self, "_param_group"):
            self._param_group = {STEP_KEY: 0}
            self._half_parameter = \
                torch.empty(numel, dtype=device_dtype).pin_memory()
            self._full_parameter = \
                torch.randn(numel, dtype=master_dtype).pin_memory()
            self._gradient = \
                torch.randn(numel, dtype=master_dtype).pin_memory()
            self._optimizer_states = \
                self._init_optimizer_states(numel, master_dtype)

        # Increment the step key.
        self._param_group[STEP_KEY] += 1

        # Conduct the optimization step.
        self._step(
            self._param_group,
            self._half_parameter, self._full_parameter,
            self._gradient, *self._optimizer_states
        )


class FlexTrainOptimizer:

    cpu_optimizer: FlexTrainCPUOptimizer = None

    """
    Abstract class for FlexTrain optimizers.
    This class mainly serves to group parameters for optimization.

    Args:
        param_groups (List[Dict]): A list where each dictionary contains
            the parameters and their respective optimization arguments.
    """
    def __init__(self, param_groups: List[Dict]):
        # Ensure that the param_groups is a list of dictionaries.
        # So that the parameters keep the same order across processes.
        assert isinstance(param_groups, Iterable)

        # Link to parameter groups.
        self.param_groups = param_groups if \
            all(isinstance(group, dict) for group in param_groups) else \
            [{PARAMS_KEY: param_groups}]  # Create a parameter group

        # Build the parameter group map.
        param_group_map: Dict[Tensor, Dict] = {}
        param_id_map: Dict[Tensor, int] = {}
        for group in param_groups:
            # Ensure parameters are stored in a list.
            assert isinstance(group[PARAMS_KEY], list)
            # Store the group for each parameter
            for param in group[PARAMS_KEY]:
                param_group_map[param] = group
                param_id_map[param] = len(param_id_map)

        # Try to group the parameters by their unit index.
        # Find non-layerwise parameters and group them.
        unit_parameter_map = get_para_coordinator().unit_parameter_map
        aligned_unit_numel = get_para_coordinator().unit_numel
        self.unit_group_segments: Dict[int, List[GroupSegment]] = {}
        non_layerwise_params = set(param_group_map.keys())

        for i, unit_paras in unit_parameter_map.items():
            paras = unit_paras.paras
            # Remove the parameters from the non-layerwise set.
            for p in paras:
                non_layerwise_params.discard(p)
            # Build the group segments for the unit parameters.
            group_segments = create_group_segments(
                paras, [param_group_map[p] for p in paras]
            )
            # Adjust the last segment to align with the unit numel.
            last_segment = group_segments[-1]
            assert last_segment.end <= aligned_unit_numel
            last_segment.end = aligned_unit_numel
            # Link the unit index to the group.
            self.unit_group_segments[i] = group_segments

        # Convert the non-layerwise parameters to a list.
        non_layerwise_params = list(non_layerwise_params)
        non_layerwise_params.sort(key=lambda p: param_id_map[p])

        # End of grouping parameters.

        # Allocate GPU memory for:
        # Gradient buffers of non-layerwise parameters for optimizer steps.
        config = get_flextrain_config()
        device_dtype = config.mixed_precision.device_dtype
        master_dtype = config.mixed_precision.master_dtype
        gradacc_dtype = config.mixed_precision.gradacc_dtype
        self._gradacc_dtype_incompatible = device_dtype != gradacc_dtype

        # Allocate gradient buffers & create / link the master parameters.
        total_numel = sum(p.numel() for p in non_layerwise_params)
        self._device_para = torch.zeros(
            total_numel, dtype=device_dtype, device=torch.cuda.current_device()
        )
        self._device_grad = torch.zeros(
            total_numel, dtype=device_dtype, device=torch.cuda.current_device()
        )
        if self._gradacc_dtype_incompatible:
            self._device_acc_grad = torch.zeros(
                total_numel, dtype=gradacc_dtype,
                device=torch.cuda.current_device()
            )
        self._master_para = torch.zeros(
            total_numel, dtype=master_dtype,
            device=torch.device('cpu'), pin_memory=True
        )
        self._master_grad = torch.zeros(
            total_numel, dtype=master_dtype,
            device=torch.device('cpu'), pin_memory=True
        )
        move_into_contiguous(non_layerwise_params, self._device_para)
        self._master_para.copy_(self._device_para)

        # Create / link the master parameters.
        offset = 0
        self.non_layerwise_master_params = []
        non_layerwise_master_grads: List[Tensor] = []

        def gradacc_hook_producer(i: int):
            def hook(grad: Tensor):
                non_layerwise_master_grads[i] += grad.flatten()
            return hook

        for i, param in enumerate(non_layerwise_params):
            end = offset + param.numel()
            # Create the master parameter.
            master_param = torch.nn.Parameter(self._master_para[offset:end])
            self.non_layerwise_master_params.append(master_param)

            # Link the gradient buffers.
            param.grad = self._device_grad[offset:end].view_as(param)
            master_param.grad = self._master_grad[offset:end]
            if self._gradacc_dtype_incompatible:
                device_master_grad = self._device_acc_grad[offset:end]
                non_layerwise_master_grads.append(device_master_grad)
                param.register_hook(gradacc_hook_producer(i))
            offset = end

        # Create a step context for the non-layerwise parameters.
        self.non_layerwise_step_contexts = create_step_contexts(
            group_segments=create_group_segments(
                non_layerwise_params,
                [param_group_map[p] for p in non_layerwise_params]
            ),
            half_parameter=None,
            full_parameter=self._master_para,
            gradient=self._master_grad,
            optimizer_states=self.cpu_optimizer._init_optimizer_states(
                total_numel, master_dtype
            )
        )

        # Detach all parameters from the coordinator.
        get_para_coordinator().detach_all_parameters()

    @property
    def optimizer_state_per_element(self) -> int:
        """ Returns the number of optimizer state per parameter element. """
        master_dtype = get_flextrain_config().mixed_precision.master_dtype
        return len(self.cpu_optimizer._init_optimizer_states(1, master_dtype))

    @property
    def non_layerwise_numel(self):
        return self._device_para.numel()

    def step(self, *args, **kwargs):
        # Perform the optimization step of non-layerwise parameters.
        device_grad_buffer = self._device_acc_grad \
            if self._gradacc_dtype_incompatible else self._device_grad

        # 1. Conduct all-reduce for the gradients of non-layerwise parameters.
        dist.all_reduce(device_grad_buffer, op=dist.ReduceOp.AVG)

        # 2. Copy the gradients to the master gradients.
        self._master_grad.copy_(device_grad_buffer, non_blocking=True)
        torch.cuda.synchronize()

        # 3. Conduct the optimization step for the non-layerwise parameters.
        self.cpu_optimizer.step(self.non_layerwise_step_contexts)
        self._device_grad.zero_()
        if self._gradacc_dtype_incompatible:
            self._device_acc_grad.zero_()

        # 4. Copy the updated master parameters back to the device.
        self._device_para.copy_(self._master_para, non_blocking=True)

    def update_states(self):
        # Update the step for each parameter group.
        for group in self.param_groups:
            # If the step key is not present, add it to the group.
            if STEP_KEY not in group:
                group[STEP_KEY] = 0
            # Increment the step key for the group.
            group[STEP_KEY] += 1
