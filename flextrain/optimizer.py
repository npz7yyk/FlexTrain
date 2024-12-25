import torch

from abc import ABC, abstractmethod
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from torch import Tensor
from typing import Iterator, List, Dict

from flextrain.config import get_flextrain_config
from flextrain.memory import FunctionHandle, move_into_contiguous
from flextrain.memory.coordinator import get_para_coordinator
from flextrain.utils import dist


@dataclass
class GroupSegment:
    start: int
    end: int
    group: Dict


def merge_segments(segments: List[GroupSegment]) -> List[GroupSegment]:
    # Remove empty segments
    segments = [seg for seg in segments if seg.start != seg.end]

    # If there are no segments, return an empty list
    if not segments:
        return []

    # Move the first segment to the start of the range
    first_segment = segments[0]
    first_segment.end -= first_segment.start
    first_segment.start = 0
    merged_segs = [first_segment]

    # Merge the segments
    for segment in segments[1:]:
        length = segment.end - segment.start
        # If the group is the same, merge the segments
        if id(merged_segs[-1].group) == id(segment.group):
            merged_segs[-1].end += length
        # Otherwise, append the segment
        else:
            segment.start = merged_segs[-1].end
            segment.end = segment.start + length
            merged_segs.append(segment)

    return merged_segs


def create_group_segments(
    params: List[Tensor], param_groups: List[Dict]
) -> List[GroupSegment]:
    group_segments = [
        GroupSegment(0, p.numel(), group)
        for p, group in zip(params, param_groups)
    ]
    group_segments = merge_segments(group_segments)
    return group_segments


def slice_segments(
    segments: List[GroupSegment], lengths: List[int]
) -> List[List[GroupSegment]]:
    """
    Slices segments into multiple segments based on the given lengths.

    Args:
        segments (List[GroupSegment]): a sorted list of contiguous segments.
        lengths (List[int]): a list of lengths to slice the segments.

    Returns:
        List[List[GroupSegment]]: a list of lists of segments that \
            reconstruct the coverage for the original segments.
    """

    results = []
    segments: Iterator[GroupSegment] = iter(segments)

    def unpack_next_segment():
        cur = next(segments)
        return cur.start, cur.end, cur.group

    # Initialize the first segment
    cur_start, cur_end, cur_group = unpack_next_segment()

    for length in lengths:
        pieces = []
        remaining = length

        # Keep slicing until we've fulfilled "length"
        while remaining > 0:
            available = cur_end - cur_start  # How much left in this segment
            if available <= remaining:
                # Use the entire leftover portion of this segment
                pieces.append(
                    GroupSegment(cur_start, cur_end, cur_group)
                )
                remaining -= available
                # Move to the next segment
                try:
                    cur_start, cur_end, cur_group = unpack_next_segment()
                except StopIteration:
                    assert remaining == 0
            else:
                # Only part of this segment is needed
                pieces.append(
                    GroupSegment(cur_start, cur_start + remaining, cur_group)
                )
                # Advance the cursor within the current segment
                cur_start += remaining
                remaining = 0  # we've fulfilled this length

        # current "length" is done -> append piece to results
        results.append(pieces)

    return results


@dataclass
class StepContext:
    group_segments: List[GroupSegment]
    parameter: Tensor
    gradient: Tensor
    optimizer_states: List[Tensor]


class FlexTrainCPUOptimizer(ABC):

    def __init__(self):
        self._ctx_opts_map: Dict[StepContext, List[Tensor]] = {}
        self._executor = ThreadPoolExecutor(max_workers=1)

    @abstractmethod
    def _init_optimizer_states(self, numel: int, dtype: torch.dtype):
        pass

    @abstractmethod
    def _step(
        self, group_args: Dict,
        parameter: Tensor, gradient: Tensor, *optimizer_states: Tensor
    ):
        pass

    @torch.no_grad()
    def step(
        self,
        step_contexts: StepContext | List[StepContext],
        async_op: bool = False
    ) -> FunctionHandle | None:
        """ Performs a single optimization step.

        Arguments:
            step_contexts (StepContext | List[StepContext]): \
                A context object that contains the target parameters, \
                gradients, and optimizer states, or a list of such objects. \
                If the context is provided without optimizer states, \
                the states will be initialized for each context instance.
            async_op (bool, optional): \
                Whether to perform the optimization step asynchronously.

        Returns:
            FunctionHandle | None: \
                A handle to the optimization step function \
                if the operation is asynchronous. Otherwise, returns None.
        """

        # Wrap the step contexts in a list if not already a list.
        if isinstance(step_contexts, StepContext):
            step_contexts = [step_contexts]

        step_funcs = []
        for step_context in step_contexts:
            # Skip empty parameter segments.
            if not step_context.parameter.numel():
                continue

            # If optimizer states are not provided,
            # check whether the optimizer states need to be initialized.
            if not step_context.optimizer_states and \
                    id(step_context) not in self._ctx_opts_map:
                self._ctx_opts_map[id(step_context)] = \
                    self._init_optimizer_states(
                        step_context.parameter.numel(),
                        step_context.parameter.dtype
                    )

            # Get all the data needed for the optimization step.
            parameter = step_context.parameter
            gradient = step_context.gradient
            optimizer_states = self._ctx_opts_map[id(step_context)] \
                if not step_context.optimizer_states \
                else step_context.optimizer_states

            # Conduct the optimization step for each group segment.
            def step_func_producer(cur_context: StepContext):
                def step_func():
                    for segment in cur_context.group_segments:
                        group_args = segment.group
                        start, end = segment.start, segment.end
                        self._step(
                            group_args,
                            parameter.data[start:end],
                            gradient.data[start:end],
                            *[state[start:end] for state in optimizer_states]
                        )
                return step_func
            step_funcs.append(step_func_producer(step_context))

        def execute_step_funcs():
            for step_func in step_funcs:
                step_func()

        # If not an asynchronous operation, execute the step function.
        if async_op:
            return FunctionHandle(
                self._executor.submit(execute_step_funcs).result
            )
        # Otherwise, execute the step function synchronously.
        else:
            execute_step_funcs()

    @torch.no_grad()
    def profile_step(self, numel: int, dtype: torch.dtype):
        # Initialize the optimizer states.
        if not hasattr(self, "_optimizer_states"):
            self._optimizer_states = self._init_optimizer_states(numel)
        # Create the parameter and gradient tensors.
        para = torch.empty(numel, dtype=dtype)
        grad = torch.empty(numel, dtype=dtype)
        # Conduct the optimization step.
        self._step(para, grad, *self._optimizer_states)


class FlexTrainOptimizer:
    """
    Abstract class for FlexTrain optimizers.
    This class mainly serves to group parameters for optimization.

    Args:
        param_groups (List[Dict]): A list where each dictionary contains
            the parameters and their respective arguments.
        opt_state_per_element (int): \
            The number of optimizer states per element in the optimizer.

    Attributes:
        param_groups (List[Dict]): A list where each dictionary contains
            the parameters and their respective arguments.
        unit_group_map (Dict): A dictionary mapping unit indices to their
            corresponding parameter groups.
        non_layerwise_params (Set): A set of parameters that do not belong
            to any unit group. These parameters will be kept in GPU memory
            and updated by the GPU.
        coordinator: An instance of the FlexTrain optimizer coordinator
            responsible for managing the optimization process.
    """

    def __init__(self, param_groups: List[Dict], opt_state_per_element: int):
        # Ensure that the param_groups is a list of dictionaries.
        # So that the parameters keep the same order across processes.
        assert isinstance(param_groups, Iterable)

        # 0. Sub-optimizers should be initialized in the derived class.
        self.cpu_optimizer: FlexTrainCPUOptimizer = None

        # 1. Link to basic parameters.
        PARAMS_KEY = "params"
        self.param_groups = param_groups if \
            all(isinstance(group, dict) for group in param_groups) else \
            [{PARAMS_KEY: param_groups}]  # Create a parameter group
        self.opt_state_per_element = opt_state_per_element

        self.param_group_map: Dict[Tensor, Dict] = {}
        param_id_map: Dict[Tensor, int] = {}
        for group in param_groups:
            # Ensure parameters are stored in a list.
            assert isinstance(group[PARAMS_KEY], list)
            # Store the group for each parameter
            for param in group[PARAMS_KEY]:
                self.param_group_map[param] = group
                param_id_map[param] = len(param_id_map)

        # 2. Try to group the parameters by their unit index.
        #    Find non-layerwise parameters and group them.
        unit_parameter_map = get_para_coordinator().unit_parameter_map
        self.unit_group_segments: Dict[int, List[GroupSegment]] = {}
        non_layerwise_params = set(self.param_group_map.keys())

        for i, unit_paras in unit_parameter_map.items():
            paras = unit_paras.paras
            # Remove the parameters from the non-layerwise set.
            for p in paras:
                non_layerwise_params.discard(p)
            # Build the group segments for the unit parameters.
            group_segments = create_group_segments(
                paras, [self.param_group_map[p] for p in paras]
            )
            # Link the unit index to the group.
            self.unit_group_segments[i] = group_segments

        # Convert the non-layerwise parameters to a list.
        non_layerwise_params = list(non_layerwise_params)
        non_layerwise_params.sort(key=lambda p: param_id_map[p])

        # End of grouping parameters.

        # 3. Allocate GPU memory for:
        #    Gradient buffers of non-layerwise parameters for optimizer steps.
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
            total_numel, dtype=master_dtype, pin_memory=True
        )
        self._master_grad = torch.zeros(
            total_numel, dtype=master_dtype, pin_memory=True
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
        self.non_layerwise_step_context = StepContext(
            group_segments=create_group_segments(
                non_layerwise_params,
                [self.param_group_map[p] for p in non_layerwise_params]
            ),
            parameter=self._master_para,
            gradient=self._master_grad,
            optimizer_states=[]
        )

        # Detach all parameters from the coordinator.
        get_para_coordinator().detach_all_parameters()

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
        self.cpu_optimizer.step(self.non_layerwise_step_context)
        self._device_grad.zero_()
        if self._gradacc_dtype_incompatible:
            self._device_acc_grad.zero_()

        # 4. Copy the updated master parameters back to the device.
        self._device_para.copy_(self._master_para, non_blocking=True)

    def submit_step(self, step_contexts: StepContext | List[StepContext]):
        # Submit an asynchronous optimization step.
        return self.cpu_optimizer.step(step_contexts, async_op=True)

    def update_state(self):
        # Update the step for each parameter group.
        STEP_KEY = "step"
        for group in self.param_groups:
            # If the step key is not present, add it to the group.
            if STEP_KEY not in group:
                group[STEP_KEY] = 0
            # Increment the step key for the group.
            group[STEP_KEY] += 1
