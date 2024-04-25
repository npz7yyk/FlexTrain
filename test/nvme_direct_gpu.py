from flextrain.memory.nvme_swapper import AsyncNVMeSwapper
from flextrain.memory import FlexTrainDataID, FlexTrainDataType
from flextrain.config import init_flextrain_config
import torch


nvme = AsyncNVMeSwapper("/shared_ssd_storage/yikang/.cache")

init_flextrain_config({
    "device_dtype": "fp16",
    "master_dtype": "fp32",
    "batch_size": 64,
    "micro_batch_size": 8,
    "micro_batch_per_block": 1,
    "num_layers": 24,
    "checkpoint_interval": 1,
    "checkpoint_split_ratio": [0.5, 0.5],
    "parameter_split_ratio": [0.5, 0.5],
    "optimizer_split_ratio": [0.5, 0.5],
    "nvme_swap_dir": "/shared_ssd_storage/yikang/.cache"
})

total = 25

for i in range(total):
    nvme.swap_out(
        FlexTrainDataID(i, FlexTrainDataType.PARA), torch.empty(1024 ** 3) 
    )

import time
from flextrain import distributed as dist

ts = []

torch.cuda.set_device(dist.get_rank())

tar = torch.empty(1024 ** 3, device=torch.cuda.current_device())
cpu1 = torch.empty(1024 ** 3, pin_memory=True)
cpu2 = torch.empty(1024 ** 3, pin_memory=True)

nvme.swap_in(
    FlexTrainDataID(0, FlexTrainDataType.PARA), cpu1
)

cpu_ava = cpu1
cpu_tar = cpu2
for i in range(total):
    t1 = time.time()
    if i < total - 1:
        h = nvme.swap_in(
            FlexTrainDataID(i + 1, FlexTrainDataType.PARA),
            cpu_tar,
            True
        )
    tar.copy_(cpu_ava)
    if i < total - 1:
        h.wait()
    cpu_ava, cpu_tar = cpu_tar, cpu_ava
    t2 = time.time()
    print(t2 - t1)
    ts.append(t2 - t1)

ts = ts[3:]
ts = ts[:-1]
print(sum(ts) / len(ts))
