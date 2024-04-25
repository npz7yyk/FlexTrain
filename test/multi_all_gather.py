import torch
import torch.distributed as dist


dist.init_process_group(backend='nccl', init_method='env://')
world_size = dist.get_world_size()
rank = dist.get_rank()
device = torch.device('cuda', rank)

# Create CUDA stream
stream = torch.cuda.Stream()

# Create events for measuring time
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

# test 2 all_gather vs 1 all_gather
scale1 = 0.0
scale2 = 1.0
size1 = int(scale1 * 1024 ** 3)
size2 = int(scale2 * 1024 ** 3)
size0 = size1 + size2

times = []

for i in range(100):
    src0 = torch.empty(size0, dtype=torch.int8, device=device)
    src1 = torch.empty(size1, dtype=torch.int8, device=device)
    src2 = torch.empty(size2, dtype=torch.int8, device=device)
    dst = torch.empty(world_size * size0, dtype=torch.int8, device=device)
    dst1, dst2 = torch.split(dst, [size1 * world_size, size2 * world_size])

    torch.cuda.synchronize()

    # Record the start event
    start_event.record(stream)

    with torch.cuda.stream(stream):
        s1 = src0[0].item()
        src2.copy_(src0, non_blocking=True)
        h1 = dist.all_gather_into_tensor(dst1, src1, async_op=True)
        h2 = dist.all_gather_into_tensor(dst2, src2, async_op=True)
        h1.wait(), h2.wait()
        s2 = dst[0].item()

    # Record the end event
    end_event.record(stream)

    # Synchronize the stream to ensure all operations are complete
    stream.synchronize()

    # Synchronize the stream to ensure all operations are complete
    if rank == 0:
        assert s1 == s2

    # Calculate the elapsed time
    elapsed_time_ms = start_event.elapsed_time(end_event)

    time_cuda = torch.tensor([elapsed_time_ms], device=device)
    dist.all_reduce(time_cuda, op=dist.ReduceOp.MAX)

    if rank == 0:
        times.append(time_cuda.item())

if rank == 0:
    times = times[10:]  # Remove the first ten times
    print("Elapsed time: {} milliseconds".format(sum(times) / len(times)))
