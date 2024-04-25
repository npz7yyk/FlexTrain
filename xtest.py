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

# test allgather preparation
scale1 = 1
size1 = int(scale1 * 1 * 1024 ** 3)
scale2 = 1
size2 = int(scale2 * 1 * 1024 ** 3)

src1 = torch.randn(size1, device=device)
dst1 = torch.randn(world_size * size1, device=device)

src2 = torch.randn(size2, device=device)
dst2 = torch.randn(world_size * size2, device=device)

torch.cuda.synchronize()

# Record the start event
start_event.record(stream)

# Perform your computation within the stream
# For example:
# torch.cuda.synchronize() # Ensure previous operations are complete before measuring time
# Your computation here
# with torch.cuda.stream(stream):
# dist.all_gather(dsts, src)
# print(dsts[0][0] + dsts[1][0] + dsts[2][0] + dsts[3][0])
h1 = dist.all_gather_into_tensor(dst1, src1, async_op=True)
h2 = dist.all_gather_into_tensor(dst2, src2, async_op=True)

h1.wait()
h2.wait()

print(dst1[0] + dst2[0])

# Record the end event
end_event.record(stream)

# Synchronize the stream to ensure all operations are complete
stream.synchronize()

# Calculate the elapsed time
elapsed_time_ms = start_event.elapsed_time(end_event)

print("Elapsed time: {} milliseconds".format(elapsed_time_ms))
