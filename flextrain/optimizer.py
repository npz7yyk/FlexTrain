import psutil
import torch

from abc import ABC, abstractmethod
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from torch import Tensor
from typing import Iterator, Tuple, List, Dict

from flextrain.config import get_flextrain_config
from flextrain.memory import (
    FunctionHandle,
    RotateContainer,
    move_into_contiguous,
    allocate_memory,
    allocate_memory_chunks
)
from flextrain.memory.coordinator import get_para_coordinator
from flextrain.utils import dist

STEP_KEY = "step"
PARAMS_KEY = "params"


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


def purify_segments(segments: List[GroupSegment]) -> List[GroupSegment]:
    """
    Creates a copy of the segments with PARAMS_KEY removed \
    from each segment.group dictionary to ensure that \
    the parameters inside are not serialized when pickling.

    Args:
        segments (List[GroupSegment]): a list of segments to purify.

    Returns:
        List[GroupSegment]: a new list of segments with PARAMS_KEY removed.
    """

    result = []
    for segment in segments:
        new_group = dict(segment.group)
        new_group.pop(PARAMS_KEY)
        result.append(GroupSegment(segment.start, segment.end, new_group))
    return result


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
    def _init_optimizer_states(
        self, numel: int, dtype: torch.dtype
    ) -> List[Tensor]:
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
            self._optimizer_states = self._init_optimizer_states(numel, dtype)
            self._param_group = {STEP_KEY: 0}
        self._param_group[STEP_KEY] += 1
        # Create the parameter and gradient tensors.
        para = torch.empty(numel, dtype=dtype)
        grad = torch.empty(numel, dtype=dtype)
        # Conduct the optimization step.
        self._step(self._param_group, para, grad, *self._optimizer_states)


class Connection:

    def send(self, obj):
        pass

    def recv(self) -> object:
        pass


# TEMPORARY: use opt_state_per_element to allocate memory for optimizer states.
# May support more complex structures in the future.
class SharedStepBuffer:
    """ Shared buffer for master parameters and optimizer states. """

    def __init__(
        self,
        master_dtype: torch.dtype,
        unit_fwd_numels: Tuple[int, int],
        unit_bwd_numels: Tuple[int, int],
        opt_state_per_element: int
    ):
        self._opt_state_per_element = opt_state_per_element

        # Structure:
        # [CPU para, NVMe para, NVMe opts1, NVMe opts2, ...]
        self._fwd_splits = [unit_fwd_numels[0]] + \
            [unit_fwd_numels[1]] * self._opt_state_per_element
        self._bwd_splits = [unit_bwd_numels[0]] + \
            [unit_bwd_numels[1]] * self._opt_state_per_element

        # Allocate memory for the shared buffer.
        max_numel = max(
            sum(self._fwd_splits), sum(self._bwd_splits)
        )
        self._buffer = allocate_memory(
            max_numel, master_dtype,
            torch.device('cpu'), pin_memory=False
        ).share_memory_()

    def get_master_parameters(self, forward: bool) -> Tensor:
        """ Get the master parameters for the unit. """
        splits = self._fwd_splits if forward else self._bwd_splits
        return self._buffer[:splits[0] + splits[1]]

    def get_cpu_master_parameters(self, forward: bool) -> Tensor:
        """ Get the CPU master parameters for the unit. """
        splits = self._fwd_splits if forward else self._bwd_splits
        return self._buffer[:splits[0]]

    def get_nvme_master_parameters(self, forward: bool) -> Tensor:
        """ Get the NVMe master parameters for the unit. """
        splits = self._fwd_splits if forward else self._bwd_splits
        return self._buffer[splits[0]:splits[0] + splits[1]]

    def get_nvme_optimizer_states(self, forward: bool) -> Tensor:
        """ Get the NVMe optimizer states for the unit. """
        splits = self._fwd_splits if forward else self._bwd_splits
        start = splits[0]
        end = sum(splits)
        return self._buffer[start:end].chunk(self._opt_state_per_element)[1:]


def step_worker_func(
    pipe: Connection,
    optimizer_class: FlexTrainCPUOptimizer,
    optimizer_args: Dict,
    master_dtype: torch.dtype,
    num_units: int,
    unit_fwd_numels: Tuple[int, int],
    unit_bwd_numels: Tuple[int, int],
    shared_grad_buffers: RotateContainer[Tensor],
    shared_step_buffers: RotateContainer[SharedStepBuffer]
):
    # Initialize the CPU optimizer.
    optimizer: FlexTrainCPUOptimizer = optimizer_class(**optimizer_args)

    # Report the memory usage.
    used_memory = psutil.Process().memory_info().rss
    pipe.send(used_memory)

    # TEMPORARY: assume all optimizer states have the same dtype.
    # May support different dtypes in the future.
    opt_state_per_element = sum([
        s.numel() for s in optimizer._init_optimizer_states(
            1, get_flextrain_config().mixed_precision.master_dtype
        )
    ])
    fwd_cpu_numel, _ = unit_fwd_numels
    bwd_cpu_numel, _ = unit_bwd_numels
    # Allocate memory for the optimizer states.
    fwd_cpu_opts = allocate_memory_chunks(
        fwd_cpu_numel, (num_units, opt_state_per_element),
        master_dtype, torch.device("cpu"), pin_memory=False
    )
    bwd_cpu_opts = allocate_memory_chunks(
        bwd_cpu_numel, (num_units, opt_state_per_element),
        master_dtype, torch.device("cpu"), pin_memory=False
    )

    # Buffer indices for shared buffers.
    INFLIGHT_BUFFER = 1

    # Wait for the parent process to send the step command.
    while True:
        cmd, *args = pipe.recv()

        # Conduct the optimization step.
        if cmd == "STEP":
            # Unpack the arguments.
            forward, unit_index, cpu_segments, nvme_segments = args
            cpu_numel, nvme_numel = unit_fwd_numels \
                if forward else unit_bwd_numels

            # Locate CPU data for optimization.
            grad_buffer = shared_grad_buffers[INFLIGHT_BUFFER]
            step_buffer = shared_step_buffers[INFLIGHT_BUFFER]

            cpu_parameter = step_buffer.get_cpu_master_parameters(forward)
            cpu_gradient = grad_buffer[:cpu_numel]
            cpu_optimizer_states = (
                fwd_cpu_opts[unit_index]
                if forward else bwd_cpu_opts[unit_index]
            ).chunk(opt_state_per_element)

            # Build CPU step context.
            cpu_step_context = StepContext(
                group_segments=cpu_segments,
                parameter=cpu_parameter,
                gradient=cpu_gradient,
                optimizer_states=cpu_optimizer_states
            )

            # Locate NVMe data for optimization.
            nvme_parameter = step_buffer.get_nvme_master_parameters(forward)
            nvme_gradient = grad_buffer[cpu_numel:cpu_numel + nvme_numel]
            nvme_optimizer_states = \
                step_buffer.get_nvme_optimizer_states(forward)

            # Build NVMe step context.
            nvme_step_context = StepContext(
                group_segments=nvme_segments,
                parameter=nvme_parameter,
                gradient=nvme_gradient,
                optimizer_states=nvme_optimizer_states
            )

            # Conduct the optimization step.
            optimizer.step([cpu_step_context, nvme_step_context])
            print(f"[StepWorker] Step {unit_index} completed.")

            # Send an acknowledgment to the parent process.
            pipe.send(f"STEP_UNIT_{unit_index}_COMPLETED")

        # Raise an error if the command is not recognized.
        else:
            # Support for other commands may be added in the future.
            # E.g., support for persisting the optimizer states.
            raise ValueError(f"Unknown command: {cmd}")


class FlexTrainOptimizer:
    """
    Abstract class for FlexTrain optimizers.
    This class mainly serves to group parameters for optimization.

    Args:
        param_groups (List[Dict]): A list where each dictionary contains
            the parameters and their respective arguments.
        optimizer_class (FlexTrainCPUOptimizer): The CPU optimizer class.
        optimizer_args (Dict): The arguments to initialize the CPU optimizer.
    """

    def __init__(
        self,
        param_groups: List[Dict],
        optimizer_class: FlexTrainCPUOptimizer,
        optimizer_args: Dict
    ):
        # Ensure that the param_groups is a list of dictionaries.
        # So that the parameters keep the same order across processes.
        assert isinstance(param_groups, Iterable)

        # Initialize the CPU optimizer.
        self.cpu_optimizer_class = optimizer_class
        self.cpu_optimizer_args = optimizer_args
        self.cpu_optimizer: FlexTrainCPUOptimizer = \
            optimizer_class(**optimizer_args)

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
                [param_group_map[p] for p in non_layerwise_params]
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

    def init_step_worker(
        self,
        num_units: int,
        unit_fwd_numels: Tuple[int, int],
        unit_bwd_numels: Tuple[int, int],
        shared_grad_buffers: RotateContainer,
        shared_step_buffers: RotateContainer
    ):
        # Initialize the step worker.
        self.parent_conn, child_conn = torch.multiprocessing.Pipe()
        self.step_worker = torch.multiprocessing.Process(
            target=step_worker_func,
            args=(
                child_conn,
                self.cpu_optimizer_class,
                self.cpu_optimizer_args,
                get_flextrain_config().mixed_precision.master_dtype,
                num_units,
                unit_fwd_numels,
                unit_bwd_numels,
                shared_grad_buffers,
                shared_step_buffers
            )
        )
        self.step_worker.daemon = True
        self.step_worker.start()

        # Get the baseline memory usage of the worker.
        self.baseline_memory = self.parent_conn.recv()

    def submit_step(
        self,
        forward: bool,
        unit_index: int,
        cpu_segments: List[GroupSegment],
        nvme_segments: List[GroupSegment]
    ) -> FunctionHandle:
        # Submit an asynchronous optimization step.
        cpu_segments = purify_segments(cpu_segments)
        nvme_segments = purify_segments(nvme_segments)
        self.parent_conn.send(
            ("STEP", forward, unit_index, cpu_segments, nvme_segments)
        )

        # Return a function handle to wait for the completion of the step.
        def wait_for_completion():
            msg = self.parent_conn.recv()
            assert msg == f"STEP_UNIT_{unit_index}_COMPLETED"
        return FunctionHandle(wait_for_completion)

    def update_state(self):
        # Update the step for each parameter group.
        for group in self.param_groups:
            # If the step key is not present, add it to the group.
            if STEP_KEY not in group:
                group[STEP_KEY] = 0
            # Increment the step key for the group.
            group[STEP_KEY] += 1
