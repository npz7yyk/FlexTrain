# Adapted from:
# deepspeed/ops/op_builder.py

import importlib
import os
import pkgutil
import sys


try:
    import flextrain.ops.builder
    # Success means runtime
    _BUILDER_DIR = "flextrain.ops.builder"
except ImportError:
    _BUILDER_DIR = "builder"

_THIS_MODULE = sys.modules[__name__]


# return an instance of op builder class, name specified by class_name
def _create_op_builder(class_name):
    if class_name in _THIS_MODULE.__dict__:
        return _THIS_MODULE.__dict__[class_name]()
    else:
        return None


# return an op builder class, name specified by class_name
def _get_op_builder(class_name):
    if class_name in _THIS_MODULE.__dict__:
        return _THIS_MODULE.__dict__[class_name]
    else:
        return None


def builder_closure(member_name):
    if _BUILDER_DIR == "flextrain.ops.builder":
        # in installation cannot get builder due to torch not installed,
        # return closure instead
        def _builder():
            return _create_op_builder(member_name)

        return _builder
    else:
        builder = _get_op_builder(member_name)
        return builder


_SKIP_MODULES = ["builder"]
_BUILDER_CLASSES = ["OpBuilder", "CPUOpBuilder", "CUDAOpBuilder"]


# reflect builder names and add builder closure
_path = [os.path.dirname(_THIS_MODULE.__file__)]
for _, module_name, _ in pkgutil.iter_modules(_path):

    # skip build-related modules
    if module_name in _SKIP_MODULES:
        continue

    module = importlib.import_module(f".{module_name}", package=_BUILDER_DIR)
    for member_name in module.__dir__():

        # Only add builder closure for builder classes
        if not member_name.endswith('Builder'):
            continue

        # Skip base builder classes
        if member_name in _BUILDER_CLASSES:
            continue

        # assign builder name to variable with same name
        _THIS_MODULE.__dict__[member_name] = builder_closure(member_name)
