# from flextrain.memory import FlexTrainDataID, FlexTrainDataTypes
# from flextrain.memory.nvme_swapper import AsyncNVMeSwapper


# nvme_swapper = AsyncNVMeSwapper("/shared_ssd_storage/yikang/.cache")


# nvme_swapper.allocated_storage(
#     FlexTrainDataID(9, FlexTrainDataTypes(FlexTrainDataTypes.CKPT)),
#     all_rank_size_in_bytes=1024**4
# )

import os
import torch
import torch.distributed as dist


dist.init_process_group(backend='nccl', init_method='env://')

rank = dist.get_rank()
print(type(dist.broadcast(torch.tensor([1], device=f"cuda:{rank}"), src=0, async_op=True)))
