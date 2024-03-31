import os
from datetime import timedelta

# Default process group wide timeout, if applicable.
# This only applies to the gloo and nccl backends
DEFAULT_PROCESS_GROUP_TIMEOUT = timedelta(
    minutes=int(os.getenv("DEEPSPEED_TIMEOUT", default=30))
)

# Default torch.distributed backend
DEFAULT_TORCH_DISTRIBUTED_BACKEND = "nccl"

# Default init_method for torch.distributed
DEFAULT_TORCH_DISTRIBUTED_INIT_METHOD = "env://"
