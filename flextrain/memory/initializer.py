import functools

from torch.nn import Parameter, Module
from typing import List

from flextrain.config import get_flextrain_config
from flextrain.memory.coordinator import get_para_coordinator


_SHUTDOWN_INIT = False


def shutdown_init_context():
    """
    Shutdown the model initializer,
    i.e. recover the original layer init temporarily.
    Has no effects if Init not used or not enabled.
    Useful when user needs a code block inside Init
    to be executed with original layer init.
    """
    global _SHUTDOWN_INIT
    _SHUTDOWN_INIT = True


def restore_init_context():
    """
    Restore the model initializer.
    Suggestion: pair with shutdown_model_initializer
    """
    global _SHUTDOWN_INIT
    _SHUTDOWN_INIT = False


class Init(object):

    def __init__(self, layer_class: type, enabled=True):
        self._enabled = enabled
        self._layer_class = layer_class

        # Init related configurations
        self._layer_per_unit = get_flextrain_config().checkpoint_interval

        # Track units
        self._unit_layers: List[Module] = []
        self._unit_count = 0

        # To avoid re-entrance
        self._exited = False

    def __enter__(self):
        assert not self._exited, (
            "Re-entry FlexTrain Init context is not allowed. "
            "This context manager should be unique for each script. "
            "Consider using shutdown_init_context() / restore_init_context() "
            "to temporarily disable and enable the context. "
        )

        if not self._enabled:
            return

        self._override_layer_init()

    def __exit__(self, *args, **kwargs):
        if not self._enabled:
            return

        # Finish potentially not finished layer management
        # Because num_layer % checkpoint_interval can be non-zero
        self._manage_unit_memory(in_exit=True)

        self._restore_layer_init()

        # Log the parameter coordinator configuration after Init
        get_para_coordinator().log_configuration()

        # Mark the exit status
        self._exited = True

    def _manage_unit_memory(self, in_exit: bool = False):
        # If the current layer is not the last layer of the unit
        # AND not in the exit context (for potentially not managed layers)
        curr_layer = len(self._unit_layers)
        if curr_layer < self._layer_per_unit and not in_exit:
            return

        # If in the exit context and there is no layer in the unit
        if in_exit and curr_layer == 0:
            return

        # Collect all the parameters in the unit
        unit_paras: List[Parameter] = []
        for layer in self._unit_layers:
            unit_paras.extend(list(layer.parameters()))

        # Manage the unit memory by the parameter coordinator
        get_para_coordinator().init_unit_parameters(
            self._unit_count, unit_paras
        )

        # Update unit information
        self._unit_layers = []
        self._unit_count += 1

    def _override_layer_init(self):
        # Save the original layer init function
        self._original_layer_init = self._layer_class.__init__

        # Start of layer_init wrapper
        @functools.wraps(self._original_layer_init)
        def _flextrain_init(module: Module, *args, **kwargs):

            # Conduct the original layer init function
            self._original_layer_init(module, *args, **kwargs)

            # If user wants to shutdown the model initializer
            # Then return immediately after original layer init
            if _SHUTDOWN_INIT:
                return

            # Track the current layer
            self._unit_layers.append(module)

            # Use the memory coordinator to manage the unit memory
            self._manage_unit_memory()

        # End of layer_init wrapper

        self._layer_class.__init__ = _flextrain_init

    def _restore_layer_init(self):
        self._layer_class.__init__ = self._original_layer_init
