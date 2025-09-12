import numpy as np
import torch

from dataclasses import dataclass
from typing import TypeAlias, Iterator, Sequence, List, Dict
ParamGroup: TypeAlias = Dict


def reshape_list(lst: List, shape: List[int]) -> np.ndarray:
    return np.asarray(lst, dtype=object).reshape(shape)


def flatten_list(nested: List) -> List:
    if not isinstance(nested, List):
        return [nested]
    result = []
    for item in nested:
        result.extend(flatten_list(item))
    return result


@dataclass
class GroupSegment:
    param_group: ParamGroup
    start: int
    end: int


@dataclass
class StepContext:
    param_group: ParamGroup
    half_parameter: torch.Tensor
    full_parameter: torch.Tensor
    gradient: torch.Tensor
    optimizer_states: List[torch.Tensor]


def is_same_param_group(pg1: ParamGroup, pg2: ParamGroup) -> bool:
    return id(pg1) == id(pg2)


def merge_segments(
    segments: List[GroupSegment],
    is_remapping: bool = False
) -> List[GroupSegment]:
    # Remove empty segments
    segments = [seg for seg in segments if seg.start != seg.end]

    # If there are no segments, return an empty list
    if not segments:
        return []

    def move_segment(segment: GroupSegment, new_offset: int) -> None:
        length = segment.end - segment.start
        segment.start = new_offset
        segment.end = new_offset + length

    # Adjust the first segment or assert it starts at 0
    first_segment = segments[0]
    if is_remapping:
        move_segment(first_segment, 0)
    else:
        assert first_segment.start == 0
    merged_segments = [first_segment]

    # Merge the segments
    for segment in segments[1:]:
        # Adjust the segment or assert continuity
        if is_remapping:
            move_segment(segment, merged_segments[-1].end)
        else:
            assert segment.start == merged_segments[-1].end
        # If the group is the same, merge the segments
        if is_same_param_group(
            segment.param_group, merged_segments[-1].param_group
        ):
            merged_segments[-1].end = segment.end
        # Otherwise, append the segment
        else:
            merged_segments.append(segment)

    return merged_segments


def create_group_segments(
    params: List[torch.Tensor], groups: List[ParamGroup]
) -> List[GroupSegment]:

    assert len(params) == len(groups)

    # Create segments for each parameter
    offset = 0
    group_segments: List[GroupSegment] = []
    for param, group in zip(params, groups):
        # Skip empty parameters
        if param.numel() == 0:
            continue

        # Create a segment for the parameter
        group_segments.append(
            GroupSegment(
                param_group=group,
                start=offset,
                end=offset + param.numel()
            )
        )
        offset += param.numel()

    # Merge the segments
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

    # If there are no segments, return len(lengths) empty lists
    if not segments:
        assert sum(lengths) == 0
        return [[] for _ in lengths]

    results: List[List[GroupSegment]] = []
    segments: Iterator[GroupSegment] = iter(segments)

    def unpack_next_segment():
        cur = next(segments)
        return cur.start, cur.end, cur.param_group

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
                    GroupSegment(cur_group, cur_start, cur_end)
                )
                remaining -= available
                # Move to the next segment
                try:
                    cur_start, cur_end, cur_group = unpack_next_segment()
                except StopIteration:
                    assert remaining == 0
                    break
            else:
                # Only part of this segment is needed
                pieces.append(
                    GroupSegment(cur_group, cur_start, cur_start + remaining)
                )
                # Advance the cursor within the current segment
                cur_start += remaining
                remaining = 0  # we've fulfilled this length

        # current "length" is done -> append piece to results
        results.append(pieces)

    return results


def create_step_contexts(
    group_segments: List[GroupSegment],
    half_parameter: torch.Tensor | None,
    full_parameter: torch.Tensor,
    gradient: torch.Tensor,
    optimizer_states: List[torch.Tensor],
) -> List[StepContext]:
    """
    Creates step contexts from the given contiguous buffers. \
    Only the group segments are used to slice the buffers.

    Args:
        group_segments (List[GroupSegment]): a list of group segments.
        half_parameter (torch.Tensor): a tensor containing all half parameters.
        full_parameter (torch.Tensor): a tensor containing all full parameters.
        gradient (torch.Tensor): a tensor containing all gradients.
        optimizer_states (List[torch.Tensor]): a list of tensors \
            containing all optimizer states.

    Returns:
        List[StepContext]: a list of step contexts.
    """

    # Create the split plans for each tensor
    numels = [gs.end - gs.start for gs in group_segments]
    numels = [n for n in numels if n > 0]

    half_parameter_splits = half_parameter.split(numels) \
        if half_parameter is not None else [None] * len(numels)
    full_parameter_splits = full_parameter.split(numels)
    gradient_splits = gradient.split(numels)
    optimizer_state_splits = [os.split(numels) for os in optimizer_states]

    # Create step contexts
    step_contexts: List[StepContext] = []
    for i, gs in enumerate(group_segments):
        step_contexts.append(
            StepContext(
                param_group=gs.param_group,
                half_parameter=half_parameter_splits[i],
                full_parameter=full_parameter_splits[i],
                gradient=gradient_splits[i],
                optimizer_states=[oss[i] for oss in optimizer_state_splits]
            )
        )

    return step_contexts


def merge_slice_plans(*plans: List[List[int]]) -> List[int]:
    """
    Merges multiple slicing plans into a single slicing plan.

    Args:
        plans (List[List[int]]): a list of slicing plans.

    Returns:
        List[int]: a merged slicing plan.
    """

    # Assert there is at least two plans to merge
    assert len(plans) >= 2

    # Merge each plan into the merged plan
    cut_points = set()
    total_length = sum(plans[0])
    for plan in plans:
        # Assert the total length is the same
        assert sum(plan) == total_length
        # Add all cut points to the set
        offset = 0
        for length in plan:
            offset += length
            cut_points.add(offset)
    # Drop the cut point 0
    cut_points.discard(0)

    # Create the merged plan from the cut points
    merged_plan = []
    sorted_cut_points = sorted(cut_points)
    prev_cut = 0
    for cut in sorted_cut_points:
        merged_plan.append(cut - prev_cut)
        prev_cut = cut

    return merged_plan


def create_tensor_split_plans(
    tensor_numels: List[int],
    merged_split_plan: List[int]
):
    """
    Create the split plans for each tensor based on a common split plan.

    Args:
        tensor_numels (List[int]): a list of tensor numels to be split.
        merged_split_plan (List[int]): a common split plan.

    Returns:
        List[List[int]]: a list of split plans for each tensor.
    """

    # Remove empty tensors
    tensor_numels = [n for n in tensor_numels if n > 0]

    # If there are no tensors, return an empty list
    if not tensor_numels:
        assert sum(merged_split_plan) == 0
        return []

    # Get the numels of each tensor
    total_numel = sum(tensor_numels)
    assert total_numel == sum(merged_split_plan)

    tensor_split_plans: List[List[int]] = []
    merged_plan_iter = iter(merged_split_plan)
    for tensor_numel in tensor_numels:
        remaining = tensor_numel
        split_plan = []
        while remaining > 0:
            length = next(merged_plan_iter)
            assert length <= remaining
            split_plan.append(length)
            remaining -= length
        tensor_split_plans.append(split_plan)

    # Assert all merged plan is consumed
    assert next(merged_plan_iter, None) is None

    return tensor_split_plans


def get_sliced_tensors(
    tensors: torch.Tensor | List[torch.Tensor],
    split_plan: List[List[int]]
) -> List[torch.Tensor]:
    if isinstance(tensors, torch.Tensor):
        tensors = [tensors]
    tensors = [t for t in tensors if t.numel() > 0]
    assert len(tensors) == len(split_plan), \
        f"Expected {[t.numel() for t in tensors]} to match {split_plan}"
    result = []
    for t, splits in zip(tensors, split_plan):
        assert t.numel() == sum(splits), \
            f"Expected {splits} to match {t.numel()}"
        result.extend(torch.split(t, splits))
    return result


def group_step_contexts(
    step_contexts: List[StepContext], split_plan: List[int]
) -> List[List[StepContext]]:
    """
    Groups step contexts based on the partition split plan.

    Args:
        step_contexts (List[StepContext]): a list of step contexts.
        split_plan (List[int]): a list of lengths to group the step contexts.

    Returns:
        List[List[StepContext]]: a list of lists of step contexts that \
            store the grouped step contexts for each partition.
    """
    grouped_contexts: List[List[StepContext]] = []
    step_contexts_iter = iter(step_contexts)

    for length in split_plan:
        remaining = length
        assert remaining > 0
        contexts: List[StepContext] = []
        while remaining > 0:
            ctx = next(step_contexts_iter)
            ctx_numel = ctx.full_parameter.numel()
            assert ctx_numel <= remaining
            contexts.append(ctx)
            remaining -= ctx_numel
        # Add the grouped contexts to the dictionary
        assert contexts
        grouped_contexts.append(contexts)

    # Assert all step contexts are consumed
    assert next(step_contexts_iter, None) is None

    return grouped_contexts


class StepContextContainer(Sequence[List[StepContext]]):
    """ An step context container that slices the task evenly. """

    def __init__(
        self,
        num_partitions: int,
        group_segments: List[GroupSegment],
        device_param_numels: List[int],
        master_param_numels: List[int],
        gradient_numels: List[int]
    ):
        # Merge the segments to ensure continuity
        group_segments = merge_segments(group_segments, is_remapping=True)
        if group_segments:
            self._empty = False
        else:
            self._empty = True
            self._grouped_step_contexts = [[] for _ in range(num_partitions)]
            return

        # Determine the total numel for slicing
        total_numel = group_segments[-1].end
        assert sum(device_param_numels) == total_numel
        assert sum(master_param_numels) == total_numel
        assert sum(gradient_numels) == total_numel
        assert total_numel % num_partitions == 0

        # Slice the segments into even parts
        ep_slice_plan = [total_numel // num_partitions] * num_partitions
        pg_slice_plan = [gs.end - gs.start for gs in group_segments]
        dp_slice_plan = device_param_numels
        mp_slice_plan = master_param_numels
        gd_slice_plan = gradient_numels
        self._merged_plan = merge_slice_plans(
            ep_slice_plan, pg_slice_plan,
            dp_slice_plan, mp_slice_plan, gd_slice_plan
        )

        # Create the split plans for each tensor
        self._ep_slice_plan = ep_slice_plan
        self._group_segments: List[GroupSegment] = \
            flatten_list(slice_segments(group_segments, self._merged_plan))
        self._half_paremter_splits = \
            create_tensor_split_plans(device_param_numels, self._merged_plan)
        self._full_parameter_splits = \
            create_tensor_split_plans(master_param_numels, self._merged_plan)
        self._gradient_splits = \
            create_tensor_split_plans(gradient_numels, self._merged_plan)

        # Grouped step contexts
        self._grouped_step_contexts: List[List[StepContext]] = []

    def plan(
        self,
        half_parameter: torch.Tensor | List[torch.Tensor],
        full_parameter: torch.Tensor | List[torch.Tensor],
        gradient: torch.Tensor | List[torch.Tensor],
        optimizer_states: List[torch.Tensor | List[torch.Tensor]]
    ):
        # If the container is empty, do nothing
        if self._empty:
            return

        # Check whether need to do extra copy
        self.extra_copy = half_parameter is not None

        # Slice each buffer according to the split plans
        self.half_parameter = get_sliced_tensors(
            half_parameter, self._half_paremter_splits
        ) if self.extra_copy else [None] * len(self._merged_plan)
        self.full_parameter = get_sliced_tensors(
            full_parameter, self._full_parameter_splits
        )
        self.gradient = get_sliced_tensors(gradient, self._gradient_splits)
        self.optimizer_states: List[List[torch.Tensor]] = []
        for state in optimizer_states:
            self.optimizer_states.append(
                get_sliced_tensors(state, self._full_parameter_splits)
            )

        # Assert the number of slices is the same
        num_slices = len(self._group_segments)
        assert len(self.half_parameter) == num_slices
        assert len(self.full_parameter) == num_slices
        assert len(self.gradient) == num_slices
        for state in self.optimizer_states:
            assert len(state) == num_slices

        # Create step contexts
        self._step_contexts: List[StepContext] = []
        for i in range(len(self._group_segments)):
            states = [state[i] for state in self.optimizer_states]
            self._step_contexts.append(
                StepContext(
                    param_group=self._group_segments[i].param_group,
                    half_parameter=self.half_parameter[i],
                    full_parameter=self.full_parameter[i],
                    gradient=self.gradient[i],
                    optimizer_states=states
                )
            )

        # Group the step contexts by partition
        self._grouped_step_contexts = \
            group_step_contexts(self._step_contexts, self._ep_slice_plan)

    def __getitem__(self, index: int) -> List[StepContext]:
        assert self._grouped_step_contexts
        assert 0 <= index < len(self._grouped_step_contexts)
        return self._grouped_step_contexts[index]

    def __len__(self) -> int:
        return len(self._grouped_step_contexts)
