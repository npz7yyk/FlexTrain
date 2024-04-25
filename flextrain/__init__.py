import torch

from argparse import ArgumentParser

import flextrain.checkpointing
import flextrain.utils.distributed as distributed

from .config import init_flextrain_config
from .defaults import (
    TORCH_DISTRIBUTED_BACKEND_DEFAULT,
    TORCH_DISTRIBUTED_INIT_METHOD_DEFAULT,
    PROCESS_GROUP_TIMEOUT_DEFAULT
)
from .engine import FlexTrainEngine
from .llm_func import LLMFuncPack
from .memory.initializer import Init
from .utils.distributed import init_distributed

# from checkpointing import (
#     data_parallel_cuda_manual_seed,
#     FWDContext,
#     detach_variable,
#     checkpointed_forward,
#     checkpointed_backward
# )


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


def initialize(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    llm_functions: LLMFuncPack,
    config: dict = None,
    dist_init_required=False,
    dist_backend=TORCH_DISTRIBUTED_BACKEND_DEFAULT,
    timeout=PROCESS_GROUP_TIMEOUT_DEFAULT,
    init_method=TORCH_DISTRIBUTED_INIT_METHOD_DEFAULT,
    rank=-1,
    world_size=-1
):
    # Initialize FlexTrain configuration if provided
    if config is not None:
        init_flextrain_config(config)

    # Initialize distributed if required
    if dist_init_required:
        distributed.init_distributed(
            dist_backend=dist_backend,
            timeout=timeout,
            init_method=init_method,
            rank=rank,
            world_size=world_size
        )

    # Wrap the model with FlexTrainEngine
    model = FlexTrainEngine(model, optimizer, llm_functions)

    return model
