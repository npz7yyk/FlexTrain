from optimizer import FlexTrainOptimizer, hash_tensor
from model_initializer import Init
from checkpointing import (
    data_parallel_cuda_manual_seed,
    FWDContext,
    detach_variable,
    checkpointed_forward,
    checkpointed_backward
)

from argparse import Namespace


# FlexTrain Enabled or Disabled
flextrain_enabled = False


def is_configured():
    """ True if FlexTrain has been configured by calling
        deepspeed.runtime.flex.is_configured, else returns false

    Arguments:
        None

    Return:
        True of configured, else False
    """
    return flextrain_enabled


def initialize(args: Namespace):
    """ Initializes FlexTrain using the configuration dictionary

    Arguments:
        args: (Namespace) deepspeed arguments to parse

    Return:
        None
    """

    assert hasattr(args, "deepspeed_config_dict"), \
        "DeepSpeed config dictionary not found in args"

    if "flex_optimization" in args.deepspeed_config_dict:
        flextrain_config_dict = args.deepspeed_config_dict["flex_optimization"]
    else:
        # FlexTrain is not configured, therefore return
        return

    # Set global flextrain_enabled
    global flextrain_enabled
    flextrain_enabled = flextrain_config_dict["enabled"]

    # If FlexTrain is not enabled, return
    if not flextrain_enabled:
        return
