import time
import torch

from tqdm import tqdm
from typing import List

from flextrain.checkpointing import detach_variable, retrieve_tensor
from flextrain.config import get_flextrain_config
from flextrain.defaults import (
    BENCHMARK_ITERATIONS,
    BENCHMARK_NUM_BLOCKS,
    BENCHMARK_BLOCK_SIZE
)
from flextrain.llm_func import LLMFunc, retrieve_llm_loss
from flextrain.memory import FlexTrainDataType
from flextrain.memory.nvme_swapper import get_nvme_swapper, FlexTrainDataID
from flextrain.utils import dist


def _record_event():
    event = torch.cuda.Event(enable_timing=True)
    event.record()
    return event


def _world_10_90_percentile_avg(nums: List[float]):
    nums.sort()
    start = int(len(nums) * 0.1)
    end = int(len(nums) * 0.9)
    avg = sum(nums[start:end]) / (end - start)
    tensor = torch.tensor(avg).cuda()
    dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
    return tensor.item()


def system_benchmark():
    # 1. Benchmark the performance of the target LLM
    pre_fwd_times, pre_bwd_times, post_times = [], [], []
    layer_fwd_times, layer_bwd_times = [], []

    iterations = tqdm(
        range(BENCHMARK_ITERATIONS), desc="LLM Perf. Benchmarking"
    ) if dist.get_rank() == 0 else range(BENCHMARK_ITERATIONS)
    for _ in iterations:
        # 1. Benchmark the forward time of pre-process
        pre_fwd_start = _record_event()
        pre_inputs, post_inputs, loss_inputs = LLMFunc.get_batch()
        first_passed_down, every_layer = LLMFunc.pre_process(pre_inputs)
        pre_fwd_end = _record_event()

        # 2. Benchmark the forward time of each layer
        passed_down = detach_variable(first_passed_down)
        unit_fwd_start = _record_event()
        unit_passed_down = LLMFunc.layer_forward(0, 1)(
            *passed_down, *every_layer
        )
        unit_fwd_end = _record_event()

        # 3. Benchmark the post-process time
        last_passed_down = detach_variable(unit_passed_down)
        post_start = _record_event()
        llm_outputs = LLMFunc.post_process(last_passed_down, post_inputs)
        llm_loss_rst = LLMFunc.loss(llm_outputs, loss_inputs)
        loss = retrieve_llm_loss(llm_loss_rst)
        torch.autograd.backward(loss)
        post_end = _record_event()

        # 4. Benchmark the backward time of each layer
        tensor = retrieve_tensor(unit_passed_down)
        gradient = torch.ones_like(tensor)
        unit_bwd_start = _record_event()
        torch.autograd.backward(tensor, gradient)
        unit_bwd_end = _record_event()

        # 5. Benchmark the backward time of pre-process
        tensor = retrieve_tensor(first_passed_down)
        gradient = torch.ones_like(tensor)
        pre_bwd_start = _record_event()
        torch.autograd.backward(tensor, gradient)
        pre_bwd_end = _record_event()

        # Record the time
        torch.cuda.synchronize()
        pre_fwd_times.append(pre_fwd_start.elapsed_time(pre_fwd_end))
        pre_bwd_times.append(pre_bwd_start.elapsed_time(pre_bwd_end))
        post_times.append(post_start.elapsed_time(post_end))
        layer_fwd_times.append(unit_fwd_start.elapsed_time(unit_fwd_end))
        layer_bwd_times.append(unit_bwd_start.elapsed_time(unit_bwd_end))

    # Compute the average time
    avg_pre_fwd = _world_10_90_percentile_avg(pre_fwd_times)
    avg_pre_bwd = _world_10_90_percentile_avg(pre_bwd_times)
    avg_post = _world_10_90_percentile_avg(post_times)
    avg_layer_fwd = _world_10_90_percentile_avg(layer_fwd_times)
    avg_layer_bwd = _world_10_90_percentile_avg(layer_bwd_times)

    dist.rank0_logger.info(
        "\n\n> FlexTrain model performance benchmarking results"
        f" (micro_batch_size={get_flextrain_config().micro_batch_size}):\n"
        f"  - Average pre-process forward time: {avg_pre_fwd:.3f} ms\n"
        f"  - Average pre-process backward time: {avg_pre_bwd:.3f} ms\n"
        f"  - Average post-process time: {avg_post:.3f} ms\n"
        f"  - Average layer forward time: {avg_layer_fwd:.3f} ms\n"
        f"  - Average layer backward time: {avg_layer_bwd:.3f} ms\n"
    )

    # 2. Conduct the benchmarking of the NVMe swapper
    swap_in_times, swap_out_times, overlap_times = [], [], []
    swapper = get_nvme_swapper()

    read_buffer = torch.empty(BENCHMARK_BLOCK_SIZE, dtype=torch.uint8)
    write_buffer = torch.empty(BENCHMARK_BLOCK_SIZE, dtype=torch.uint8)

    for i in range(BENCHMARK_NUM_BLOCKS):
        write_buffer += 1
        swapper.swap_out(
            FlexTrainDataID(FlexTrainDataType.PARA, i), write_buffer
        )

    # Benchmark the swap-in time
    iterations = tqdm(
        range(BENCHMARK_ITERATIONS), desc="NVMe Read Benchmarking"
    ) if dist.get_rank() == 0 else range(BENCHMARK_ITERATIONS)
    for i in iterations:
        read_index = i % BENCHMARK_NUM_BLOCKS
        swap_in_start = time.time()
        swapper.swap_in(
            FlexTrainDataID(FlexTrainDataType.PARA, read_index), read_buffer
        )
        swap_in_end = time.time()
        swap_in_times.append(swap_in_end - swap_in_start)

    # Benchmark the swap-out time
    iterations = tqdm(
        range(BENCHMARK_ITERATIONS), desc="NVMe Write Benchmarking"
    ) if dist.get_rank() == 0 else range(BENCHMARK_ITERATIONS)
    for i in iterations:
        write_index = i % BENCHMARK_NUM_BLOCKS
        swap_out_start = time.time()
        swapper.swap_out(
            FlexTrainDataID(FlexTrainDataType.PARA, write_index), write_buffer
        )
        swap_out_end = time.time()
        swap_out_times.append(swap_out_end - swap_out_start)

    # Benchmark the overlap time
    iterations = tqdm(
        range(BENCHMARK_ITERATIONS), desc="NVMe Read-Write Benchmarking"
    ) if dist.get_rank() == 0 else range(BENCHMARK_ITERATIONS)
    for i in iterations:
        read_index = (i + 1) % BENCHMARK_NUM_BLOCKS
        write_index = i % BENCHMARK_NUM_BLOCKS
        overlap_start = time.time()
        swapper.swap_in(
            FlexTrainDataID(FlexTrainDataType.PARA, read_index),
            read_buffer, async_op=True
        )
        swapper.swap_out(
            FlexTrainDataID(FlexTrainDataType.PARA, write_index),
            write_buffer, async_op=True
        )
        swapper.synchronize()
        overlap_end = time.time()
        overlap_times.append(overlap_end - overlap_start)

    # Compute the average time
    avg_swap_in = _world_10_90_percentile_avg(swap_in_times)
    avg_swap_out = _world_10_90_percentile_avg(swap_out_times)
    avg_overlap = _world_10_90_percentile_avg(overlap_times)
    # Compute the bandwidth
    swap_in = BENCHMARK_BLOCK_SIZE / 1024 ** 3 / avg_swap_in
    swap_out = BENCHMARK_BLOCK_SIZE / 1024 ** 3 / avg_swap_out
    overlap = BENCHMARK_BLOCK_SIZE / 1024 ** 3 / avg_overlap

    world_size = dist.get_world_size()
    dist.rank0_logger.info(
        "\n\n> FlexTrain NVMe swapper benchmarking results:\n"
        f"  - Average swap-in bandwidth: {world_size} x {swap_in:.3f} GB/s\n"
        f"  - Average swap-out bandwidth: {world_size} x {swap_out:.3f} GB/s\n"
        f"  - Average overlap bandwidth: {world_size} x {overlap:.3f} GB/s\n"
    )

    # Exit the script
    exit()
