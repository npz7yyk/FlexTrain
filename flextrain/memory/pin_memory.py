import math
import torch

from typing import Tuple


def allocate_optimal(n: int, size: int):
    """
    Find the optimal way to allocate n blocks of 'size' bytes
    using only powers-of-two chunk sizes.

    Returns:
        total_alloc (int)  : minimal total allocated memory
        allocation_plan (dict): mapping chunk_size -> number_of_chunks used
    """
    # 1. k1: smallest power of 2 >= size
    #    k2: smallest power of 2 >= n * size
    def smallest_power_of_two_at_least(x):
        k = 0
        while (1 << k) < x:
            k += 1
        return k

    k1 = smallest_power_of_two_at_least(size)
    k2 = smallest_power_of_two_at_least(n * size)

    # Build a list of chunk sizes (powers-of-two) and their capacities
    chunk_options = []
    for k in range(k1, k2 + 1):
        chunk_size = 1 << k  # 2^k
        # capacity = how many blocks fit in chunk_size
        capacity = math.floor(chunk_size / size)
        if capacity > 0:
            chunk_options.append((chunk_size, capacity))

    # 2. DP for minimal allocated memory to store exactly x blocks.
    #    dp[x] = minimal total allocated memory for x blocks
    dp = [float('inf') for _ in range(n + 1)]
    #    Keep track of "choice" for reconstruction.
    #    choice[x] = (chunk_size, capacity) that led to dp[x]
    choice = [None] * (n + 1)

    # no blocks => 0 allocated memory
    dp[0] = 0.0

    for x in range(1, n + 1):
        for (chunk_size, capacity) in chunk_options:
            if capacity <= 0:
                continue
            prev_x = x - capacity
            if prev_x < 0:
                # If capacity is bigger than x,
                # store x blocks with 1 chunk of this size
                prev_x = 0
            # candidate cost
            candidate_cost = dp[prev_x] + chunk_size
            if candidate_cost < dp[x]:
                dp[x] = candidate_cost
                choice[x] = (chunk_size, capacity)

    # 3. Reconstruct the plan from 'choice'
    allocation_plan = {}
    x = n
    while x > 0 and choice[x] is not None:
        chunk_size, capacity = choice[x]

        # Count how many of this chunk size are used
        allocation_plan[chunk_size] = allocation_plan.get(chunk_size, 0) + 1

        x -= capacity
        x = max(0, x)

    return allocation_plan


def reshape_list(flat_list, shape):
    """ Reshape a 1-D list into a nested list of dimensions. """
    # Base case: if shape is 1-D, return the list directly
    if len(shape) == 1:
        return flat_list

    # Otherwise, chunk the list along the first dimension
    sub_shape_size = math.prod(shape[1:])

    nested_list = []
    start_index = 0

    for _ in range(shape[0]):
        # slice out the chunk for this row (or block)
        chunk = flat_list[start_index: start_index + sub_shape_size]
        start_index += sub_shape_size

        # reshape the chunk to the sub-shape
        sub_list = reshape_list(chunk, shape[1:])
        nested_list.append(sub_list)

    return nested_list


def allocate_pin_memory_blocks(
    numel: int,
    dtype: torch.dtype,
    chunks: Tuple[int, ...]
):
    """ Allocate numel elements of dtype in pinned memory.

    Args:
        numel (int): total number of elements to allocate
        dtype (torch.dtype): data type of the tensor
        chunks (Tuple[int, ...]): the shape of the chunks to be managed in list

    Returns:
        blocks: list of tensors of shape chunks
    """

    # Wrap the chunks in a tuple
    if isinstance(chunks, int):
        chunks = (chunks,)
    block_remaining = math.prod(chunks)

    # If numel == 0, return empty tensors
    if numel == 0:
        all_blocks = [
            torch.empty(numel, dtype=dtype, pin_memory=True)
            for _ in range(block_remaining)
        ]
        return reshape_list(all_blocks, chunks)

    # Each block is byte_per_block bytes
    element_size = torch.empty([], dtype=dtype).element_size()
    byte_per_block = numel * element_size

    # Find the optimal allocation plan
    plan = allocate_optimal(block_remaining, byte_per_block)

    # Allocate the blocks
    all_blocks = []
    for chunk_bytes, num_chunks in plan.items():
        if block_remaining <= 0:
            break

        for _ in range(num_chunks):
            if block_remaining <= 0:
                break

            # Allocate one big buffer as int8, pinned memory
            chunk_buffer = torch.empty(
                chunk_bytes, dtype=torch.int8, pin_memory=True
            )

            # Calculate how many blocks can be fit in this chunk
            capacity_per_chunk = chunk_bytes // byte_per_block

            usable_blocks = min(block_remaining, capacity_per_chunk)
            for i in range(usable_blocks):
                offset = i * byte_per_block

                # Slice a sub-tensor of size = block_size_bytes (still int8)
                subtensor = chunk_buffer[offset: offset + byte_per_block]

                # View as the final dtype
                subtensor = subtensor.view(dtype=dtype)

                # Append to the list
                all_blocks.append(subtensor)

            block_remaining -= usable_blocks

    assert block_remaining == 0, "Failed to allocate all blocks"

    return reshape_list(all_blocks, chunks)
