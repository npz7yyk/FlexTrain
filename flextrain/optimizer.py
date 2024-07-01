from abc import ABC, abstractmethod
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass
from torch import Tensor
from typing import List, Dict

from flextrain.config import get_flextrain_config
from flextrain.memory.coordinator import (
    get_model_coordinator,
    get_opt_coordinator
)


PARAMS_KEY = "params"
STEP_KEY = "step"


class FlexTrainOptimizer:
    """
    Abstract class for FlexTrain optimizers.
    This class mainly serves to group parameters for optimization.

    Args:
        param_groups (List[Dict]): A list where each dictionary contains
            the parameters and their respective arguments.
        each_numel_num_states (int): The number of states for each parameter.

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

    def __init__(self, param_groups: List[Dict], each_numel_num_states):
        # 1. Get the arguments for each parameter.
        assert all(isinstance(group, dict) for group in param_groups), (
            "FlexTrain optimizer needs parameter groups "
            "rather than a list of parameters."
        )
        self.param_groups = param_groups

        self.param_group_map = {}
        for group in param_groups:
            # Store the group for each parameter
            for param in group[PARAMS_KEY]:
                self.param_group_map[param] = group

        # 2. Try to group the parameters by their unit index.
        #    Find non-layerwise parameters and group them.
        unit_parameter_map = get_model_coordinator().unit_parameter_map
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
                continue
            # Link the unit index to the group.
            self.unit_group_map[i] = group

        # End of grouping parameters.

        # 3. Initialize the optimizer coordinator.
        self.coordinator = get_opt_coordinator()
        self.coordinator.initialize(
            self.unit_group_map.keys(),
            each_numel_num_states
        )

    def is_cpu_optimizer_needed(self):
        return get_flextrain_config().split_ratio.optimizer[0] < 1

    @property
    def gpu_layerwise_states(self):
        # Note: it is not necessary to keep the unit order.
        return [
            self.coordinator.get_gpu_unit_states(unit)
            for unit in self.unit_group_map.keys()
        ]


@dataclass
class OptTar(ABC):
    para: Tensor
    grad: Tensor


class FlexTrainCPUOptimizer(ABC):

    def __init__(self, unit_group_map: Dict[int, Dict]):
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._future: Future = None
        self._unit_group_map = unit_group_map

        # Initialize the state for each unit.
        if not hasattr(self, "state"):
            self.state = defaultdict(dict)
        for unit in unit_group_map:
            self.state[unit] = {STEP_KEY: 0}

    @abstractmethod
    def unit_step(self, step: int, args: Dict, opt_tar: OptTar):
        pass

    def synchronize(self):
        if self._future is None:
            return
        self._future.result()
        self._future = None

    def submit_unit_step(self, unit_index: int, opt_tar: OptTar):
        assert unit_index in self._unit_group_map, (
            "The unit index is not in the unit group map."
        )

        # Submit the step function to the executor.
        self._future = self._executor.submit(
            self.unit_step,
            self.state[unit_index][STEP_KEY],
            self._unit_group_map[unit_index],
            opt_tar
        )

        # Update the step count.
        self.state[unit_index][STEP_KEY] += 1
