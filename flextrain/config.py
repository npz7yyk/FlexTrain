import torch
from typing import Tuple

from flextrain.defaults import (
    AUTO_CONFIG, AUTO_CONFIG_DEFAULT,
    CHECKPOINT_INTERVAL, CHECKPOINT_INTERVAL_DEFAULT,
    SPLIT_RATIO,
    CHECKPOINT, CHECKPOINT_DEFAULT,
    GRADIENT, GRADIENT_DEFAULT,
    PARAMETER, PARAMETER_DEFAULT,
    OPTIMIZER, OPTIMIZER_DEFAULT,
    ALPHA, ALPHA_DEFAULT,
    NVME_SWAP,
    SWAP_DIR, SWAP_DIR_DEFAULT,
    AIO_BLOCK_SIZE, AIO_BLOCK_SIZE_DEFAULT,
    AIO_QUEUE_DEPTH, AIO_QUEUE_DEPTH_DEFAULT,
    AIO_THREAD_COUNT, AIO_THREAD_COUNT_DEFAULT,
    AIO_SINGLE_SUBMIT, AIO_SINGLE_SUBMIT_DEFAULT,
    AIO_OVERLAP_EVENTS, AIO_OVERLAP_EVENTS_DEFAULT,
    MIXED_PRECISION,
    DEVICE_DTYPE, DEVICE_DTYPE_DEFAULT,
    GRADACC_DTYPE, GRADACC_DTYPE_DEFAULT,
    MASTER_DTYPE, MASTER_DTYPE_DEFAULT,
    DYNAMIC_LOSS_SCALING, DYNAMIC_LOSS_SCALING_DEFAULT,
    INITIAL_SCALE_POWER, INITIAL_SCALE_POWER_DEFAULT,
    LOSS_SCALE_WINDOW, LOSS_SCALE_WINDOW_DEFAULT,
    HYSTERESIS, HYSTERESIS_DEFAULT,
    CONSECUTIVE_HYSTERESIS, CONSECUTIVE_HYSTERESIS_DEFAULT,
    MIN_LOSS_SCALE, MIN_LOSS_SCALE_DEFAULT
)
from flextrain.utils import rank0_logger, LEFT_BRACE, RIGHT_BRACE


class SplitRatioConfig:
    """
    Hyperparameters for data split ratio.
    """

    checkpoint: Tuple[float, float]
    """
    How to split the checkpointed activations among the memory hierarchy.
    Ratio = (GPU, CPU), and NVMe = 1 - GPU - CPU.
    Defaults to (1.0, 0.0) if not provided.
    """

    gradient: Tuple[float, float]
    """
    How to split the gradients of activations among the memory hierarchy.
    Note: These gradients are NOT those of the model parameters.
    Ratio = (GPU, CPU), GPU + CPU = 1.0 is required.
    Defaults to (1.0, 0.0) if not provided.
    """

    parameter: Tuple[float, float]
    """
    How to split the parameters among the memory hierarchy.
    Ratio = (GPU, CPU), and NVMe = 1 - GPU - CPU.
    Defaults to (1.0, 0.0) if not provided.
    """

    optimizer: Tuple[float, float]
    """
    How to split the optimizer states among the memory hierarchy.
    Note: FP32 model parameters are also considered as optimizer states.
    Ratio = (GPU, CPU), and NVMe = 1 - GPU - CPU.
    Defaults to (0.0, 1.0) if not provided.
    """

    alpha: float
    """
    How to split the CPU optimizer task between forward and backward.
    Ratio = forward, backward = 1 - forward.
    Defaults to 0.25 if not provided.
    """

    def __init__(self, split_ratio: dict):
        # Assertions.
        assert isinstance(split_ratio, dict), \
            "Split ratio configuration must be provided as a dictionary."

        # Set split ratio.
        self.checkpoint = split_ratio.get(CHECKPOINT, CHECKPOINT_DEFAULT)
        self.gradient = split_ratio.get(GRADIENT, GRADIENT_DEFAULT)
        self.parameter = split_ratio.get(PARAMETER, PARAMETER_DEFAULT)
        self.optimizer = split_ratio.get(OPTIMIZER, OPTIMIZER_DEFAULT)
        self.alpha = [split_ratio.get(ALPHA, ALPHA_DEFAULT)]

    def to_log(self, indent=0):
        tab_indent = "\t" * indent
        return (
            f"{tab_indent}\"{SPLIT_RATIO}\": {LEFT_BRACE}\n"
            f"{tab_indent}\t\"{CHECKPOINT}\": {self.checkpoint},\n"
            f"{tab_indent}\t\"{GRADIENT}\": {self.gradient},\n"
            f"{tab_indent}\t\"{PARAMETER}\": {self.parameter},\n"
            f"{tab_indent}\t\"{OPTIMIZER}\": {self.optimizer}\n"
            f"{tab_indent}\t\"{ALPHA}\": {self.alpha}\n"
            f"{tab_indent}{RIGHT_BRACE}"
        )


class NVMeSwapConfig:
    """
    Hyperparameters for NVMe swap.
    """

    swap_dir: str
    """
    Directory for NVMe swap files.
    Defaults to `CURRENT_DIR/nvme_swap` if not provided.
    """

    aio_block_size: int
    """
    Any tensor smaller than this size will not be swapped out.
    Defaults to 1MB (1048576 bytes) if not provided.
    """

    aio_queue_depth: int
    """
    Default number of aio requests to keep in flight.
    Defaults to 8 if not provided.
    """

    aio_thread_count: int
    """
    Default number of aio threads.
    Defaults to 1 if not provided.
    """

    aio_single_submit: bool
    """
    Default whether to submit aio requests one at a time.
    Defaults to False if not provided.
    """

    aio_overlap_events: bool
    """
    Default whether to overlap aio events.
    Defaults to True if not provided.
    """

    def __init__(self, nvme_swap: dict):
        # Assertions.
        assert isinstance(nvme_swap, dict), \
            "NVMe swap configuration must be provided as a dictionary."

        # Set NVMe swap.
        self.swap_dir = nvme_swap.get(
            SWAP_DIR, SWAP_DIR_DEFAULT)
        self.aio_block_size = nvme_swap.get(
            AIO_BLOCK_SIZE, AIO_BLOCK_SIZE_DEFAULT)
        self.aio_queue_depth = nvme_swap.get(
            AIO_QUEUE_DEPTH, AIO_QUEUE_DEPTH_DEFAULT)
        self.aio_thread_count = nvme_swap.get(
            AIO_THREAD_COUNT, AIO_THREAD_COUNT_DEFAULT)
        self.aio_single_submit = nvme_swap.get(
            AIO_SINGLE_SUBMIT, AIO_SINGLE_SUBMIT_DEFAULT)
        self.aio_overlap_events = nvme_swap.get(
            AIO_OVERLAP_EVENTS, AIO_OVERLAP_EVENTS_DEFAULT)

    def to_log(self, indent=0):
        tab_indent = "\t" * indent
        return (
            f"{tab_indent}\"{NVME_SWAP}\": {LEFT_BRACE}\n"
            f"{tab_indent}\t\"{SWAP_DIR}\": \"{self.swap_dir}\",\n"
            f"{tab_indent}\t\"{AIO_BLOCK_SIZE}\": {self.aio_block_size},\n"
            f"{tab_indent}\t\"{AIO_QUEUE_DEPTH}\": {self.aio_queue_depth},\n"
            f"{tab_indent}\t\"{AIO_THREAD_COUNT}\": {self.aio_thread_count},\n"
            f"{tab_indent}\t\"{AIO_SINGLE_SUBMIT}\": "
            f"{self.aio_single_submit},\n"
            f"{tab_indent}\t\"{AIO_OVERLAP_EVENTS}\": "
            f"{self.aio_overlap_events}\n"
            f"{tab_indent}{RIGHT_BRACE}"
        )


class MixedPercisionConfig:
    """
    Hyperparameters for mixed precision.
    """

    device_dtype: torch.dtype
    """
    Data type used for accelerator in mixed precision training.
    Defaults to torch.float16 if not provided.
    """

    gradacc_dtype: torch.dtype
    """
    Data type used for gradient accumulation in mixed precision training.
    Defaults to torch.float32 if not provided.
    """

    master_dtype: torch.dtype
    """
    Data type used for master copy in mixed precision training.
    Defaults to torch.float32 if not provided.
    """

    dynamic_loss_scaling: bool
    """
    Whether to use dynamic loss scaling for mixed precision training.
    Defaults to True (use dynamic loss scaling) if not provided.
    """

    initial_scale_power: int
    """
    Initial loss scale power for dynamic loss scaling.
    i.e. initial_loss_scale = 2 ** initial_scale_power.
    Defaults to 16 if not provided.
    """

    loss_scale_window: int
    """
    The Window size over which to raise/lower the dynamic loss scale value.
    Defaults to 1000 if not provided.
    """

    hysteresis: int
    """
    Hysteresis value for the delay shift in dynamic loss scaling.
    Defaults to 2 if not provided.
    """

    consecutive_hysteresis: bool
    """
    whether to refill the hysteresis if we reach an iteration
    that doesn't overflow. Defaults to False if not provided.
    """

    min_loss_scale: float
    """
    Minimum loss scale value for dynamic loss scaling.
    Defaults to 1.0 if not provided.
    """

    def __init__(self, mixed_precision: dict):
        # Assertions.
        assert isinstance(mixed_precision, dict), \
            "Mixed precision configuration must be provided as a dictionary."

        # Set mixed precision settings.
        self.device_dtype = mixed_precision.get(
            DEVICE_DTYPE, DEVICE_DTYPE_DEFAULT)
        self.gradacc_dtype = mixed_precision.get(
            GRADACC_DTYPE, GRADACC_DTYPE_DEFAULT)
        self.master_dtype = mixed_precision.get(
            MASTER_DTYPE, MASTER_DTYPE_DEFAULT)
        self.dynamic_loss_scaling = mixed_precision.get(
            DYNAMIC_LOSS_SCALING, DYNAMIC_LOSS_SCALING_DEFAULT)
        self.initial_scale_power = mixed_precision.get(
            INITIAL_SCALE_POWER, INITIAL_SCALE_POWER_DEFAULT)
        self.loss_scale_window = mixed_precision.get(
            LOSS_SCALE_WINDOW, LOSS_SCALE_WINDOW_DEFAULT)
        self.hysteresis = mixed_precision.get(
            HYSTERESIS, HYSTERESIS_DEFAULT)
        self.consecutive_hysteresis = mixed_precision.get(
            CONSECUTIVE_HYSTERESIS, CONSECUTIVE_HYSTERESIS_DEFAULT)
        self.min_loss_scale = mixed_precision.get(
            MIN_LOSS_SCALE, MIN_LOSS_SCALE_DEFAULT)

        def _convert_dtype(dtype):
            if isinstance(dtype, torch.dtype):
                return dtype
            # Assert dtype is a string.
            assert isinstance(dtype, str), \
                f"Dtype {dtype} must be a string if not a torch.dtype"
            # Remove "torch." prefix if present.
            if "torch." in dtype:
                dtype = dtype.replace("torch.", "")
            if dtype in ["fp16", "half", "float16"]:
                return torch.float16
            elif dtype in ["fp32", "float", "float32"]:
                return torch.float32
            elif dtype in ["bf16", "bfloat16"]:
                return torch.bfloat16
            else:
                raise ValueError(f"Dtype {dtype} not supported by FlexTrain")

        # Convert dtypes to torch.dtype.
        self.device_dtype = _convert_dtype(self.device_dtype)
        self.gradacc_dtype = _convert_dtype(self.gradacc_dtype)
        self.master_dtype = _convert_dtype(self.master_dtype)

    def to_log(self, indent=0):
        tab_indent = "\t" * indent
        return (
            f"{tab_indent}\"{MIXED_PRECISION}\": {LEFT_BRACE}\n"
            f"{tab_indent}\t\"{DEVICE_DTYPE}\": \"{self.device_dtype}\",\n"
            f"{tab_indent}\t\"{GRADACC_DTYPE}\": \"{self.gradacc_dtype}\",\n"
            f"{tab_indent}\t\"{MASTER_DTYPE}\": \"{self.master_dtype}\",\n"
            f"{tab_indent}\t\"{DYNAMIC_LOSS_SCALING}\": "
            f"{self.dynamic_loss_scaling},\n"
            f"{tab_indent}\t\"{INITIAL_SCALE_POWER}\": "
            f"{self.initial_scale_power},\n"
            f"{tab_indent}\t\"{LOSS_SCALE_WINDOW}\": "
            f"{self.loss_scale_window},\n"
            f"{tab_indent}\t\"{HYSTERESIS}\": {self.hysteresis},\n"
            f"{tab_indent}\t\"{CONSECUTIVE_HYSTERESIS}\": "
            f"{self.consecutive_hysteresis},\n"
            f"{tab_indent}\t\"{MIN_LOSS_SCALE}\": {self.min_loss_scale}\n"
            f"{tab_indent}{RIGHT_BRACE}"
        )


class FlexTrainConfig:
    """
    Hyperparameters for FlexTrain optimizations.
    """
    auto_config: bool
    """
    Whether to automatically configure FlexTrain optimizations.
    Defaults to False if not provided.
    """

    batch_size: int
    """
    Batch size required by training algorithm.
    MUST be provided by user.
    """

    micro_batch_size: int
    """
    Micro batch size.
    Batch size must be divisible by world size x micro_batch_size.
    MUST be provided by user.
    """

    checkpoint_interval: int
    """
    Checkpoint interval used for activation checkpointing.
    Defaults to 1 if not provided.
    """

    split_ratio: SplitRatioConfig
    """
    Data split ratio configuration for FlexTrain optimizations.
    If not provided, defaults to (1.0, 0.0) for all levels.
    """

    nvme_swap: NVMeSwapConfig
    """
    NVMe swap configuration for FlexTrain optimizations.
    Defaults to None if not needed, but MUST be provided if necessary.
    """

    mixed_precision: MixedPercisionConfig
    """
    Mixed precision configuration for FlexTrain optimizations.
    Defaults values will be used if not provided.
    """

    _NECESSARY_KEYS = ["batch_size", "micro_batch_size"]

    def __init__(self, config_dict: dict):
        # Assertions.
        assert isinstance(config_dict, dict), \
            "FlexTrain configuration must be provided as a dictionary."
        assert all(key in config_dict for key in self._NECESSARY_KEYS), (
            f"FlexTrain configuration must contain key {self._NECESSARY_KEYS}."
        )

        # Set necessary keys.
        self.batch_size = config_dict["batch_size"]
        self.micro_batch_size = config_dict["micro_batch_size"]

        # Set remaining keys.
        self.auto_config = config_dict.get(
            AUTO_CONFIG, AUTO_CONFIG_DEFAULT
        )
        self.checkpoint_interval = config_dict.get(
            CHECKPOINT_INTERVAL, CHECKPOINT_INTERVAL_DEFAULT
        )

        # Set split ratio.
        self.split_ratio = SplitRatioConfig(
            split_ratio=config_dict.get(SPLIT_RATIO, {})
        )

        # Set NVMe swap.
        self.nvme_swap = NVMeSwapConfig(
            nvme_swap=config_dict.get(NVME_SWAP, {})
        )

        # Set mixed precision.
        self.mixed_precision = MixedPercisionConfig(
            mixed_precision=config_dict.get(MIXED_PRECISION, {})
        )

        if self.auto_config:
            # Log FlexTrain configuration for auto-config mode.
            rank0_logger.info(
                f"\n\n> FlexTrain conducting auto-configuration ...\n"
                f"Test configuration: {LEFT_BRACE}\n"
                f"\t\"batch_size\": {self.batch_size},\n"
                f"\t\"micro_batch_size\": {self.micro_batch_size},\n"
                f"\t\"{CHECKPOINT_INTERVAL}\": {self.checkpoint_interval},\n"
                f"{self.nvme_swap.to_log(indent=1)},\n"
                f"{self.mixed_precision.to_log(indent=1)}\n"
                f"{RIGHT_BRACE}\n"
            )
        else:
            # Log FlexTrain configuration.
            rank0_logger.info(
                f"\n\n> FlexTrain configuration: {LEFT_BRACE}\n"
                f"\t\"batch_size\": {self.batch_size},\n"
                f"\t\"micro_batch_size\": {self.micro_batch_size},\n"
                f"\t\"{CHECKPOINT_INTERVAL}\": {self.checkpoint_interval},\n"
                f"{self.split_ratio.to_log(indent=1)},\n"
                f"{self.nvme_swap.to_log(indent=1)},\n"
                f"{self.mixed_precision.to_log(indent=1)}\n"
                f"{RIGHT_BRACE}\n"
            )


_CONFIG = None


def init_flextrain_config(config_dict):
    """
    Initialize FlexTrain configuration using configuration dictionary.
    If FlexTrain configuration is not found, return None.
    If FlexTrain is not enabled, return None.

    Args:
        config_dict (dict): FlexTrain configuration dictionary

    Returns:
        None
    """

    # Set _CONFIG to FlexTrain configuration.
    global _CONFIG
    _CONFIG = FlexTrainConfig(config_dict)


def get_flextrain_config():
    """
    Get FlexTrain configuration. Assert that configuration is initialized.

    Returns:
        FlexTrainConfig: FlexTrain configuration
    """

    assert _CONFIG is not None, "FlexTrain configuration not initialized"
    return _CONFIG
