import torch

from dataclasses import dataclass


@dataclass
class FlexTrainConfig:
    """
    Sets parameters for FlexTrain optimizations.
    """

    dtype: torch.dtype
    """
    Data type used for forward and backward passes.
    """

    world_size: int
    """
    World size used for distributed training.
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

    micro_batch_per_block: int
    """
    Micro batch per block.
    """

    num_layers: int
    """
    Number of layers in the given LLM.
    """

    checkpoint_interval: int
    """
    Checkpoint interval used for activation checkpointing.
    """


_CONFIG = None


def _config_key_check(flex_config_dict):
    """
    Check for the required keys in FlexTrain configuration dictionary.
    If any of the required keys are missing, raise a ValueError.

    Args:
        flex_config_dict (dict): FlexTrain configuration dictionary

    Returns:
        None
    """

    required_keys = [
        "dtype",
        "batch_size",
        "micro_batch_size",
        "micro_batch_per_block",
        "num_layers",
        "checkpoint_interval"
    ]

    for key in required_keys:
        if key not in flex_config_dict:
            raise ValueError(f"FlexTrain configuration missing key: {key}")


def _config_set_dtype(flex_config_dict):
    """
    Convert dtype in configuration dictionary from string to torch.dtype.
    If dtype is already a torch.dtype, do nothing.

    Args:
        flex_config_dict (dict): FlexTrain configuration dictionary

    Returns:
        None
    """

    assert "dtype" in flex_config_dict, \
        "FlexTrain configuration missing key: dtype"

    dtype = flex_config_dict["dtype"]
    if isinstance(dtype, torch.dtype):
        return

    if dtype in ["fp16", "half", "float16", "torch.float16"]:
        flex_config_dict["dtype"] = torch.float16
    elif dtype in ["fp32", "float", "float32", "torch.float32"]:
        flex_config_dict["dtype"] = torch.float32
    elif dtype in ["bf16", "bfloat16", "torch.bfloat16"]:
        flex_config_dict["dtype"] = torch.bfloat16
    else:
        raise ValueError(f"Dtype {dtype} not supported by FlexTrain")


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

    # Check for the required keys in configuration dictionary.
    _config_key_check(config_dict)

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
