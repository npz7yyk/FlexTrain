import torch

from torch import Tensor
from typing import List, Dict

from flextrain.config import get_flextrain_config
from flextrain.memory.coordinator import (
    get_para_coordinator,
    get_opt_coordinator,
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
                    assert id(group) == id(self.param_group_map[p]), \
                        "All parameters in a unit should be in the same group."
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

        # End of grouping parameters.

        # 3. Allocate GPU memory for:
        #    Gradient buffers of non-layerwise parameters for optimizer steps.
        config = get_flextrain_config()
        device_dtype = config.mixed_precision.device_dtype
        master_dtype = config.mixed_precision.master_dtype

        # Allocate gradient buffers & create / link the master parameters.
        total_numel = sum(p.numel() for p in self.non_layerwise_params)
        self._device_grads = torch.zeros(
            total_numel, dtype=device_dtype, device=torch.cuda.current_device()
        )
        self._master_grads = torch.zeros(
            total_numel, dtype=master_dtype, device=torch.cuda.current_device()
        )

        # Create / link the master parameters.
        offset = 0
        for param in self.non_layerwise_params:
            param.data = param.data.to(
                device=torch.cuda.current_device(), dtype=device_dtype
            )
            master_param = torch.nn.Parameter(param.to(master_dtype))
            end = offset + param.numel()
            param.grad = self._device_grads[offset:end].view_as(param)
            master_param.grad = self._master_grads[offset:end].view_as(param)
            offset = offset + param.numel()
        # End of allocating GPU memory for non-layerwise gradients.

        # 4. Initialize the optimizer coordinator.
        coordinator = get_opt_coordinator()
        coordinator.initialize(self.cpu_optimizer, self.opt_state_per_element)

    def is_cpu_optimizer_needed(self):
        return get_flextrain_config().split_ratio.optimizer[0] < 1

    def step(self, closure=None):
        # Perform the optimization step of non-layerwise parameters.

        # 1. Conduct all-reduce for the gradients of non-layerwise parameters.
        dist.all_reduce(self._device_grads, op=dist.ReduceOp.AVG)

        # 2. Conduct the optimization step for the non-layerwise parameters.
        # self.gpu_optimizer.step(closure)
        self.gpu_optimizer.zero_grad(set_to_none=False)

    def update_state(self):
        self.cpu_optimizer.update_state()
