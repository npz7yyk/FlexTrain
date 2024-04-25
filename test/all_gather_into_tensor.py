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

# test all_gather vs all_gather_into_tensor
scale = 1.00
size = int(scale * 1024 ** 3)

times = []

for i in range(100):
    src = torch.empty(size, dtype=torch.int8, device=device)
    dst = torch.empty(world_size * size, dtype=torch.int8, device=device)
    dsts = [torch.empty(size, dtype=torch.int8, device=device) for _ in range(world_size)]

    torch.cuda.synchronize()

    # Record the start event
    start_event.record(stream)

    with torch.cuda.stream(stream):
        # h1 = dist.all_gather(dsts, src, async_op=True)
        h = dist.all_gather_into_tensor(dst, src, async_op=True)
        h.wait()

    # Record the end event
    end_event.record(stream)

    # Synchronize the stream to ensure all operations are complete
    stream.synchronize()

    # Calculate the elapsed time
    elapsed_time_ms = start_event.elapsed_time(end_event)

    time_cuda = torch.tensor([elapsed_time_ms], device=device)
    dist.all_reduce(time_cuda, op=dist.ReduceOp.MAX)

    if rank == 0:
        times.append(time_cuda.item())

if rank == 0:
    times = times[10:]  # Remove the first ten times
    print("Elapsed time: {} milliseconds".format(sum(times) / len(times)))
