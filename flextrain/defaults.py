import os
from datetime import timedelta


# --- distributed defaults ---

# Default process group wide timeout, if applicable.
# This only applies to the gloo and nccl backends
PROCESS_GROUP_TIMEOUT_DEFAULT = timedelta(
    minutes=int(os.getenv("FLEXTRAIN_TIMEOUT", default=30))
)

# Default torch.distributed backend
TORCH_DISTRIBUTED_BACKEND_DEFAULT = "nccl"

# Default init_method for torch.distributed
TORCH_DISTRIBUTED_INIT_METHOD_DEFAULT = "env://"

# --- end of distributed defaults ---


# --- async io defaults ---

# Any tensor smaller than this size will not be swapped out
AIO_BLOCK_SIZE_DEFAULT = 1048576

# Default number of aio requests to keep in flight
AIO_QUEUE_DEPTH_DEFAULT = 8

# Default number of aio threads
AIO_THREAD_COUNT_DEFAULT = 1

# Default whether to submit aio requests one at a time
AIO_SINGLE_SUBMIT_DEFAULT = False

# Default whether to overlap aio events
AIO_OVERLAP_EVENTS_DEFAULT = True

# --- end of async io defaults ---
