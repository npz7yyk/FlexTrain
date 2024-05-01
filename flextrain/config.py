import torch

from dataclasses import dataclass
from typing import Tuple


@dataclass
class FlexTrainConfig:
    """
    Sets parameters for FlexTrain optimizations.
    """

    device_dtype: torch.dtype
    """
    Mixed precision data type used for accelerator.
    """

    master_dtype: torch.dtype
    """
    Mixed precision data type used for master copy.
    """

    batch_size: int
    """
    Batch size required by training algorithm.
    """

    micro_batch_size: int
    """
    Micro batch size.
    Batch size must be divisible by world size x micro_batch_size.
    """

    num_layers: int
    """
    Number of layers in the given LLM.
    """

    checkpoint_interval: int
    """
    Checkpoint interval used for activation checkpointing.
    """

    checkpoint_split_ratio: Tuple[float, float]
    """
    How to split the checkpointed activations among the memory hierarchy.
    Ratio = (GPU, CPU), and NVMe = 1 - GPU - CPU.
    User can also provide numels of each level as:
    Ratio = (GPU, CPU), and NVMe = TOTAL - GPU - CPU.
    """

    parameter_split_ratio: Tuple[float, float]
    """
    How to split the parameters among the memory hierarchy.
    Ratio = (GPU, CPU), and NVMe = 1 - GPU - CPU.
    User can also provide numels of each level as:
    Ratio = (GPU, CPU), and NVMe = TOTAL - GPU - CPU.
    """

    optimizer_split_ratio: Tuple[float, float]
    """
    How to split the optimizer states among the memory hierarchy.
    Ratio = (GPU, CPU), and NVMe = 1 - GPU - CPU.
    User can also provide numels of each level as:
    Ratio = (GPU, CPU), and NVMe = TOTAL - GPU - CPU.
    Note: FP32 model parameters are also considered as optimizer states.
    """

    nvme_swap_dir: str
    """
    Directory for NVMe swap files.
    Must be provided if NVMe swap is necessary for training.
    """


_CONFIG = None


def _config_set_dtype(flex_config_dict):
    """
    Convert dtype in configuration dictionary from string to torch.dtype.
    If dtype is already a torch.dtype, do nothing.

    Args:
        flex_config_dict (dict): FlexTrain configuration dictionary

    Returns:
        None
    """

    assert "device_dtype" in flex_config_dict, \
        "FlexTrain configuration missing key: device_dtype"
    assert "master_dtype" in flex_config_dict, \
        "FlexTrain configuration missing key: master_dtype"

    def convert_dtype(dtype):
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

    # Convert device_dtype and master_dtype.
    device_dtype = flex_config_dict["device_dtype"]
    flex_config_dict["device_dtype"] = convert_dtype(device_dtype)
    master_dtype = flex_config_dict["master_dtype"]
    flex_config_dict["master_dtype"] = convert_dtype(master_dtype)


def init_flextrain_config(config_dict):
    """
    Initialize FlexTrain configuration using configuration dictionary.
    If FlexTrain configuration is not found, return None.
    If FlexTrain is not enabled, return None.

    Args:
        flex_config_dict (dict): DeepSpeed configuration dictionary

    Returns:
        None
    """

    # Conduct type conversion.
    _config_set_dtype(config_dict)

    # Set _CONFIG to FlexTrain configuration.
    global _CONFIG
    _CONFIG = FlexTrainConfig(**config_dict)


def get_flextrain_config():
    """
    Get FlexTrain configuration. Assert that configuration is initialized.

    Returns:
        FlexTrainConfig: FlexTrain configuration
    """

    assert _CONFIG is not None, "FlexTrain configuration not initialized"
    return _CONFIG
