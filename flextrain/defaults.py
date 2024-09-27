import os
import torch

from datetime import timedelta

# --- flextrain configuration defaults ---

# # --- general configuration defaults ---
# Name of the checkpoint interval configuration
CHECKPOINT_INTERVAL = "checkpoint_interval"
# Default checkpoint interval
CHECKPOINT_INTERVAL_DEFAULT = 1

# # --- split ratio configuration defaults ---

# Name of the split ratio configuration
SPLIT_RATIO = "split_ratio"

# Name of the checkpoint split ratio configuration
CHECKPOINT = "checkpoint"
# Default checkpoint split ratio
CHECKPOINT_DEFAULT = (1.0, 0.0)

# Name of the gradient split ratio configuration
GRADIENT = "gradient"
# Default gradient split ratio
GRADIENT_DEFAULT = (1.0, 0.0)

# Name of the parameter split ratio configuration
PARAMETER = "parameter"
# Default parameter split ratio
PARAMETER_DEFAULT = (1.0, 0.0)

# Name of the optimizer split ratio configuration
OPTIMIZER = "optimizer"
# Default optimizer split ratio
OPTIMIZER_DEFAULT = (1.0, 0.0)

# # --- end of split ratio configuration defaults ---


# # --- NVMe swap configuration defaults ---

# Name of the NVMe swap configuration
NVME_SWAP = "nvme_swap"

# Name of the NVMe swap directory configuration
SWAP_DIR = "swap_dir"
# Default NVMe swap directory
SWAP_DIR_DEFAULT = os.path.join(os.getcwd(), "nvme_swap")

# Name of the NVMe swap block size configuration
AIO_BLOCK_SIZE = "aio_block_size"
# Any tensor smaller than this size will not be swapped out
AIO_BLOCK_SIZE_DEFAULT = 1048576

# Name of the NVMe swap queue depth configuration
AIO_QUEUE_DEPTH = "aio_queue_depth"
# Default number of aio requests to keep in flight
AIO_QUEUE_DEPTH_DEFAULT = 8

# Name of the NVMe swap thread count configuration
AIO_THREAD_COUNT = "aio_thread_count"
# Default number of aio threads
AIO_THREAD_COUNT_DEFAULT = 1

# Name of the NVMe swap single submit configuration
AIO_SINGLE_SUBMIT = "aio_single_submit"
# Default whether to submit aio requests one at a time
AIO_SINGLE_SUBMIT_DEFAULT = False

# Name of the NVMe swap overlap events configuration
AIO_OVERLAP_EVENTS = "aio_overlap_events"
# Default whether to overlap aio events
AIO_OVERLAP_EVENTS_DEFAULT = True

# # --- end of NVMe swap configuration defaults ---


# # --- Mixed Precision configuration defaults ---

# Name of the Mixed Precision configuration
MIXED_PRECISION = "mixed_precision"

# Name of the device dtype configuration
DEVICE_DTYPE = "device_dtype"
# Default device dtype
DEVICE_DTYPE_DEFAULT = torch.float16

# Name of the master dtype configuration
MASTER_DTYPE = "master_dtype"
# Default master dtype
MASTER_DTYPE_DEFAULT = torch.float32

# Name of the dynamic loss scaling configuration
DYNAMIC_LOSS_SCALING = "dynamic_loss_scaling"
# Default whether to use dynamic loss scaling
DYNAMIC_LOSS_SCALING_DEFAULT = True

# Name of the initial loss scale configuration
INITIAL_SCALE_POWER = "initial_scale_power"
# Default initial loss scale
INITIAL_SCALE_POWER_DEFAULT = 16

# Name of the loss scale window configuration
LOSS_SCALE_WINDOW = "loss_scale_window"
# Default loss scale window
LOSS_SCALE_WINDOW_DEFAULT = 1000

# Name of the loss scale shift hysteresis configuration
HYSTERESIS = "hysteresis"
# Default loss scale shift hysteresis
HYSTERESIS_DEFAULT = 2

# Name of the refill hysterisis configuration
CONSECUTIVE_HYSTERESIS = "consecutive_hysteresis"
# Default refill hysterisis
CONSECUTIVE_HYSTERESIS_DEFAULT = False

# Name of the min loss scale configuration
MIN_LOSS_SCALE = "min_loss_scale"
# Default min loss scale
MIN_LOSS_SCALE_DEFAULT = 1.0

# # --- end of Mixed Precision configuration defaults ---

# --- end of flextrain configuration defaults ---


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
