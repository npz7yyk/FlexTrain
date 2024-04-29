import torch
import torch.distributed as dist

import time

# from torch.cuda import _sanitizer
# _sanitizer.enable_cuda_sanitizer()


dist.init_process_group(backend='nccl', init_method='env://')
world_size = dist.get_world_size()
rank = dist.get_rank()
device = torch.device('cuda', rank)

# Create CUDA stream
stream = torch.cuda.Stream()
stream2 = torch.cuda.Stream()

# Create events for measuring time
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

# test all_gather vs all_gather_into_tensor
scale = 2
size = int(scale * 1024 ** 3)

times = []

gpu_new1 = torch.empty(size, dtype=torch.int8, device=device).fill_(rank + 0)
gpu_new2 = torch.empty(size, dtype=torch.int8, device=device)

gpu_src, gpu_new = gpu_new1, gpu_new2

for i in range(100):
    cpu_src = torch.empty(size, dtype=torch.int8, pin_memory=True).fill_(rank + i + 1)
    gpu_tar = torch.empty(size * world_size, dtype=torch.int8, device=device)
    gpu_slf = torch.split(gpu_tar, size)[rank]

    torch.cuda.synchronize()

    # Record the start event
    # start_event.record()
    # t1 = time.time()

    if i == 5:
        end2ent_t1 = time.time()

    with torch.cuda.stream(stream):
        # gpu_new.copy_(cpu_src, non_blocking=True)
        handle = dist.all_gather_into_tensor(
            # gpu_tar, gpu_src
            gpu_tar, gpu_slf.copy_(cpu_src, non_blocking=True),
            async_op=True
        )

    # with torch.cuda.stream(stream2):
    #     gpu_new.copy_(cpu_src, non_blocking=True)

    # Record the end event
    # end_event.record()

    # Synchronize the stream to ensure all operations are complete
    # torch.cuda.synchronize()
    # handle.wait()
    torch.cuda.current_stream().wait_stream(stream)

    # gpu_src, gpu_new = gpu_new, gpu_src

    # t2 = time.time()

    # Calculate the elapsed time
    # elapsed_time_ms = start_event.elapsed_time(end_event)
    elapsed_time_ms = 0
    # elapsed_time_ms = (t2 - t1) * 1000

    if rank == 0:
        gpu_srcs = gpu_tar.split(size)
        for j in range(world_size):
            assert torch.all(gpu_srcs[j] == j + i + 1).item()

    time_cuda = torch.tensor([elapsed_time_ms], device=device)
    dist.all_reduce(time_cuda, op=dist.ReduceOp.MAX)

    if rank == 0:
        times.append(time_cuda.item())
        if i % 5 == 0:
            print("Iteration: {}, Time: {} milliseconds".format(i, time_cuda.item()))

if rank == 0:
    times = times[10:]  # Remove the first ten times
    print("Elapsed time: {} milliseconds".format(sum(times) / len(times)))
    end2ent_t2 = time.time()
    print("End-to-end time: {} seconds".format(end2ent_t2 - end2ent_t1))
