import os
import torch

from datetime import timedelta

# --- flextrain configuration defaults ---

# # --- general configuration defaults ---

# Name of the system auto-config option
AUTO_CONFIG = "auto_config"
# Default whether to auto-configure the system
AUTO_CONFIG_DEFAULT = False

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
CHECKPOINT_DEFAULT = (0.0, 1.0)

# Name of the gradient split ratio configuration
GRADIENT = "gradient"
# Default gradient split ratio
GRADIENT_DEFAULT = (0.0, 1.0)

# Name of the parameter split ratio configuration
PARAMETER = "parameter"
# Default parameter split ratio
PARAMETER_DEFAULT = (0.0, 1.0)

# Name of the optimizer split ratio configuration
OPTIMIZER = "optimizer"
# Default optimizer split ratio
OPTIMIZER_DEFAULT = (0.0, 1.0)

# Name of the optimizer step split ratio configuration
ALPHA = "alpha"
# Default optimizer step split ratio
ALPHA_DEFAULT = 0.25

# # --- end of split ratio configuration defaults ---


# # --- Benchmark configuration defaults ---

# Number of iterations to run GPU related benchmarks
BENCHMARK_GPU_ITERATIONS = 200

# Number of iterations to run PCIe related benchmarks
BENCHMARK_PCIE_ITERATIONS = 200

# Number of iterations to run NVMe related benchmarks
BENCHMARK_NVME_ITERATIONS = 50

# Number of iterations to run CPU related benchmarks
BENCHMARK_CPU_ITERATIONS = 50

# Size of the data blocks to test for data traffic
BENCHMARK_TRAFFIC_BLOCK_SIZE = 2 ** 32

# Size of the data blocks to test for optimizer step
BENCHMARK_OPTIMIZER_BLOCK_SIZE = 2 ** 28

# Number of data blocks to test
BENCHMARK_NUM_BLOCKS = 16

# # --- end of Benchmark configuration defaults ---


# # --- Auto-config configuration defaults ---

# Ratio of the accessible GPU memory to the total GPU memory
ACCESSIBLE_GPU_MEMORY_RATIO = 0.95

# Ratio of the accessible CPU memory to the total CPU memory
ACCESSIBLE_CPU_MEMORY_RATIO = 1.00

# Potential alpha values to test
POTENTIAL_ALPHA_VALUES = [0.01 * i for i in range(31)]

# If above this threshold, the throughput is considered to be better
THROUGHPUT_STABLE_THRESHOLD = 1.01

# A small penalty for CPU activation gradient / NVMe data
REGULARIZATION_PENALTY = 1e-8

# Number of extra configurations to collect after the best configuration
NUM_EXTRA_CONFIGS = 20

# # --- end of Auto-config configuration defaults ---


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
AIO_THREAD_COUNT_DEFAULT = 4

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

# Name of the gradient accumulation dtype configuration
GRADACC_DTYPE = "gradacc_dtype"
# Default gradient accumulation dtype
GRADACC_DTYPE_DEFAULT = torch.float32

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
