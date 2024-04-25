import torch
import time

total = 100


ts = []

scale = 2

tar0 = torch.empty(4 * scale * 1024 ** 3, device=torch.cuda.current_device())
tar1, tar2, tar3, tar4 = torch.chunk(tar0, 4)



for i in range(total):
    cpu0 = torch.empty(4 * scale * 1024 ** 3, pin_memory=True)
    cpu1 = torch.empty(scale * 1024 ** 3, pin_memory=True)
    cpu2 = torch.empty(scale * 1024 ** 3, pin_memory=True)
    t1 = time.time()
    # tar1.copy_(cpu1, non_blocking=True)
    # tar2.copy_(cpu2, non_blocking=True)
    tar0.copy_(cpu0, non_blocking=True)
    torch.cuda.synchronize()
    t2 = time.time()
    print(t2 - t1)
    ts.append(t2 - t1)

ts = ts[3:]
ts = ts[:-1]
print(sum(ts) / len(ts))
