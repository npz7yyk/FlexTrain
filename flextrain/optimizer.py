import torch

from torch import Tensor
from typing import List, Dict

from flextrain.config import get_flextrain_config
from flextrain.memory import move_into_contiguous
from flextrain.memory.coordinator import get_para_coordinator
from flextrain.memory.coordinator.optimizer import (  # noqa: F401
    OptTarget,
    FlexTrainCPUOptimizer
)
from flextrain.utils import dist


PARAMS_KEY = "params"


class FlexTrainOptimizer:
    """
    Abstract class for FlexTrain optimizers.
    This class mainly serves to group parameters for optimization.

    Args:
        param_groups (List[Dict]): A list where each dictionary contains
            the parameters and their respective arguments.
        opt_state_per_element (int): \
            The number of optimizer states per element in the optimizer.

    Attributes:
        param_groups (List[Dict]): A list where each dictionary contains
            the parameters and their respective arguments.
        unit_group_map (Dict): A dictionary mapping unit indices to their
            corresponding parameter groups.
        non_layerwise_params (Set): A set of parameters that do not belong
            to any unit group. These parameters will be kept in GPU memory
            and updated by the GPU.
        coordinator: An instance of the FlexTrain optimizer coordinator
            responsible for managing the optimization process.
    """

    def __init__(self, param_groups: List[Dict], opt_state_per_element: int):
        # Ensure that the param_groups is a list of dictionaries.
        # So that the parameters keep the same order across processes.
        assert isinstance(param_groups, list)

        # 0. Sub-optimizers should be:
        self.cpu_optimizer: FlexTrainCPUOptimizer = None
        self.gpu_optimizer: torch.optim.Optimizer = None

        # 1. Get the arguments for each parameter.
        assert all(isinstance(group, dict) for group in param_groups), (
            "FlexTrain optimizer needs parameter groups "
            "rather than a list of parameters."
        )
        self.param_groups = param_groups
        self.opt_state_per_element = opt_state_per_element

        self.param_group_map: Dict[Tensor, Dict] = {}
        param_id_map: Dict[Tensor, int] = {}
        for group in param_groups:
            # Ensure parameters are stored in a list.
            assert isinstance(group[PARAMS_KEY], list)
            # Store the group for each parameter
            for param in group[PARAMS_KEY]:
                self.param_group_map[param] = group
                param_id_map[param] = len(param_id_map)

        # 2. Try to group the parameters by their unit index.
        #    Find non-layerwise parameters and group them.
        unit_parameter_map = get_para_coordinator().unit_parameter_map
        self.unit_group_map: Dict[int, Dict] = {}
        self.non_layerwise_params = set(self.param_group_map.keys())

        for i, unit_paras in unit_parameter_map.items():
            params = unit_paras.paras
            group = None
            # Check:
            # for each unit, all parameters of it are in the same group.
            # If the whole unit is not under training, skip it.
            for p in params:
                # Remove the parameter from the non-layerwise set.
                self.non_layerwise_params.discard(p)

                # Probably not all parameters are under training.
                if p not in self.param_group_map:
                    assert group is None, (
                        "All parameters in a unit should be "
                        "all under training or all not under training."
                    )
                    continue

                if group is None:
                    group = self.param_group_map[p]
                else:
                    pass
                    # FlexTrain will support this in the future.
                    # assert id(group) == id(self.param_group_map[p]), \
                    #     "Parameters in a unit should be in the same group."
            # The whole unit is not under training.
            if group is None:
                assert NotImplementedError, (
                    "FlexTrain currently only supports training "
                    "all the parameters in the model."
                )
            # Link the unit index to the group.
            self.unit_group_map[i] = group

        # Convert the non-layerwise parameters to a list.
        self.non_layerwise_params = list(self.non_layerwise_params)
        self.non_layerwise_params.sort(key=lambda p: param_id_map[p])
        self.non_layerwise_param_groups = [
            self.param_group_map[p] for p in self.non_layerwise_params
        ]

        # End of grouping parameters.

        # 3. Allocate GPU memory for:
        #    Gradient buffers of non-layerwise parameters for optimizer steps.
        config = get_flextrain_config()
        device_dtype = config.mixed_precision.device_dtype
        master_dtype = config.mixed_precision.master_dtype

        # Allocate gradient buffers & create / link the master parameters.
        total_numel = sum(p.numel() for p in self.non_layerwise_params)
        self._device_para = torch.zeros(
            total_numel, dtype=device_dtype, device=torch.cuda.current_device()
        )
        self._device_grad = torch.zeros(
            total_numel, dtype=device_dtype, device=torch.cuda.current_device()
        )
        self._master_para = torch.zeros(
            total_numel, dtype=master_dtype, pin_memory=True
        )
        self._master_grad = torch.zeros(
            total_numel, dtype=master_dtype, pin_memory=True
        )
        move_into_contiguous(self.non_layerwise_params, self._device_para)
        self._master_para.copy_(self._device_para)

        # Create / link the master parameters.
        offset = 0
        self.non_layerwise_master_params = []
        for param in self.non_layerwise_params:
            end = offset + param.numel()
            # Create the master parameter.
            master_param = torch.nn.Parameter(self._master_para[offset:end])
            self.non_layerwise_master_params.append(master_param)

            # Link the gradient buffers.
            param.grad = self._device_grad[offset:end].view_as(param)
            master_param.grad = self._master_grad[offset:end]
            offset = end

    def step(self, closure=None):
        # Perform the optimization step of non-layerwise parameters.

        # 1. Conduct all-reduce for the gradients of non-layerwise parameters.
        dist.all_reduce(self._device_grad, op=dist.ReduceOp.AVG)
        # Will supports master_dtype gradient accumulation in the future.

        # 2. Copy the gradients to the master gradients.
        self._master_grad.copy_(self._device_grad, non_blocking=True)
        torch.cuda.synchronize()

        # 3. Conduct the optimization step for the non-layerwise parameters.
        self.cpu_optimizer.step(closure)
        self._device_grad.zero_()

        # 4. Copy the updated master parameters back to the device.
        self._device_para.copy_(self._master_para, non_blocking=True)

    def update_state(self):
        self.cpu_optimizer.update_state()
