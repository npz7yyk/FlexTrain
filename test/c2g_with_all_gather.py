import torch
import torch.distributed as dist
import time


dist.init_process_group(backend='nccl', init_method='env://')
world_size = dist.get_world_size()
rank = dist.get_rank()
device = torch.device('cuda', rank)

# test all_gather vs all_gather_into_tensor
scale = 0.5
size = int(scale * 1024 ** 3)

cpu_src = torch.empty(size, dtype=torch.int8, pin_memory=True).fill_(rank)
gpu_tar = torch.empty(size * world_size, dtype=torch.int8, device=device)
gpu_slf = torch.split(gpu_tar, size)[rank]

end2ent_t1 = time.time()

for i in range(100):
    dist.all_gather_into_tensor(
        gpu_tar, gpu_slf.copy_(cpu_src)
    )
    torch.cuda.synchronize()

if rank == 0:
    end2ent_t2 = time.time()
    print("End-to-end time: {} seconds".format(end2ent_t2 - end2ent_t1))
