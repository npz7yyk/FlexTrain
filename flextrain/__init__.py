from argparse import ArgumentParser

from .config import init_flextrain_config
# from .model_initializer import Init
from .utils import distributed
from .utils.distributed import init_distributed
# from .optimizer import LLMFuncPack

# from checkpointing import (
#     data_parallel_cuda_manual_seed,
#     FWDContext,
#     detach_variable,
#     checkpointed_forward,
#     checkpointed_backward
# )


# FlexTrain Enabled or Disabled
flextrain_enabled = False


def add_config_arguments(parser: ArgumentParser) -> ArgumentParser:
    """
    Update the argument parser to enabling FlexTrain args parsing.
    The set of FlexTrain arguments include the following:
    1) --flextrain: boolean flag to enable FlexTrain
    2) --flextrain-config <json file path>: path of a json configuration

    Arguments:
        parser: argument parser
    Return:
        parser: Updated Parser
    """
    group = parser.add_argument_group("FlexTrain", "FlexTrain configurations")

    group.add_argument(
        "--flextrain",
        default=False,
        action="store_true",
        help="Enable FlexTrain"
    )

    group.add_argument(
        "--flextrain-config",
        default=None,
        type=str,
        help="Path to FlexTrain json configuration."
    )

    return parser


# def is_configured():
#     """ True if FlexTrain has been configured by calling
#         FlexTrain.runtime.flex.is_configured, else returns false

#     Arguments:
#         None

#     Return:
#         True of configured, else False
#     """
#     return flextrain_enabled


# def initialize(args: Namespace):
#     """ Initializes FlexTrain using the configuration dictionary

#     Arguments:
#         args: (Namespace) FlexTrain arguments to parse

#     Return:
#         None
#     """

#     assert hasattr(args, "FlexTrain_config_dict"), \
#         "FlexTrain config dictionary not found in args"

#     if "flex_optimization" in args.FlexTrain_config_dict:
#         flextrain_config_dict = args.FlexTrain_config_dict["flex_optimization"]
#     else:
#         # FlexTrain is not configured, therefore return
#         return

#     # Set global flextrain_enabled
#     global flextrain_enabled
#     flextrain_enabled = flextrain_config_dict["enabled"]

#     # If FlexTrain is not enabled, return
#     if not flextrain_enabled:
#         return
