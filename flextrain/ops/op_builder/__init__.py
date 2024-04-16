# Parts of the code here are adapted from DeepSpeed
#
# Repository: https://github.com/microsoft/DeepSpeed
# File: deepspeed/ops/op_builder/__init__.py
# Commit: 0d9cfa0
# License: Apache-2.0

import importlib
import os
import pkgutil
import sys


# List of all available op builders from flextrain op_builder
try:
    import flextrain.ops.op_builder  # noqa: F401 # type: ignore
    op_builder_dir = "flextrain.ops.op_builder"
except ImportError:
    op_builder_dir = "op_builder"

__op_builders__ = []

_this_module = sys.modules[__name__]


# Lazy initialization of op_builder class dictionary
# Stores all valid <class name, class type> mapping
_OP_BUILDER_CLASS_DICT = None

_AVOID_MODULES = ["builder"]
_AVOID_CLASSES = ["OpBuilder", "CUDAOpBuilder", "TorchCPUOpBuilder"]


def _lazy_init_class_dict():
    global _OP_BUILDER_CLASS_DICT

    # if already initialized, return
    if _OP_BUILDER_CLASS_DICT is not None:
        return

    # begin initialize for create_op_builder()
    _OP_BUILDER_CLASS_DICT = {}

    # put all valid <class name, class type> mapping into class_dict
    op_builder_module = importlib.import_module(op_builder_dir)
    op_builder_absolute_path = os.path.dirname(op_builder_module.__file__)
    for _, module_name, _ in pkgutil.iter_modules([op_builder_absolute_path]):
        # avoid self references
        if module_name in _AVOID_MODULES:
            continue

        # skip sub_directories which contains ops for other backend.
        if os.path.isdir(os.path.join(op_builder_absolute_path, module_name)):
            continue

        module = importlib.import_module(f"{op_builder_dir}.{module_name}")
        for member_name in module.__dir__():
            # avoid abstract classes
            if member_name in _AVOID_CLASSES:
                continue

            if member_name.endswith('Builder'):
                if member_name not in _OP_BUILDER_CLASS_DICT:
                    _OP_BUILDER_CLASS_DICT[member_name] = \
                        getattr(module, member_name)
    # end initialize for create_op_builder()


def _create_op_builder(class_name):
    _lazy_init_class_dict()
    if class_name in _OP_BUILDER_CLASS_DICT:
        return _OP_BUILDER_CLASS_DICT[class_name]()
    else:
        return None


def _get_op_builder(class_name):
    _lazy_init_class_dict()
    if class_name in _OP_BUILDER_CLASS_DICT:
        return _OP_BUILDER_CLASS_DICT[class_name]
    else:
        return None


def _builder_closure(member_name):
    if op_builder_dir == "op_builder":
        # during installation time cannot get builder due to
        # torch not installed, return closure instead
        def _builder():
            builder = _create_op_builder(member_name)
            return builder

        return _builder
    else:
        # during runtime, return op builder class directly
        builder = _get_op_builder(member_name)
        return builder


# Dynamically load all op builders
_abs_op_builder_dir = os.path.dirname(os.path.abspath(__file__))
for _, module_name, _ in pkgutil.iter_modules([_abs_op_builder_dir]):
    if module_name in _AVOID_MODULES:
        continue

    module = importlib.import_module(f".{module_name}", package=op_builder_dir)
    for member_name in module.__dir__():
        if member_name in _AVOID_CLASSES:
            continue

        if member_name.endswith('Builder'):
            # assign builder name to variable with same name
            _this_module.__dict__[member_name] = _builder_closure(member_name)
