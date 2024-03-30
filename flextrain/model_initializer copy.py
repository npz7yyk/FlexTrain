import functools
import os
import torch

from contextlib import nullcontext
from torch import Tensor
from torch.nn import Parameter, Module
from typing import Callable, Iterable


_original_torch_tensor = torch.tensor
_original_torch_empty = torch.empty
_original_torch_zeros = torch.zeros
_original_torch_ones = torch.ones
_original_torch_full = torch.full
_original_torch_arange = torch.arange
_original_torch_eye = torch.eye
_original_torch_randn = torch.randn


def get_new_tensor_fn_for_dtype(dtype: torch.dtype) -> Callable:

    def new_tensor(_class, *args, **kwargs) -> Tensor:
        device_index = os.environ["LOCAL_RANK"]
        device = torch.device(f"cuda:{device_index}")
        tensor = _original_torch_empty(0, device=device).new_empty(*args, **kwargs)
        if tensor.is_floating_point():
            tensor = tensor.to(dtype)

        return tensor

    return new_tensor


def fp_tensor_constructor_wrapper(
    func: Callable,
    target_fp_dtype: torch.dtype
) -> Callable:

    def wrapped_func(*args, **kwargs) -> Tensor:
        if kwargs.get("device", None) is None:
            device_index = os.environ["LOCAL_RANK"]
            kwargs["device"] = torch.device(f"cuda:{device_index}")
        tensor: Tensor = func(*args, **kwargs)
        if tensor.is_floating_point():
            tensor.data = tensor.data.to(target_fp_dtype)

        return tensor

    return wrapped_func


def get_all_subclasses(root_class):
    subclass_list = []

    def recurse(_class):
        for subclass in _class.__subclasses__():
            subclass_list.append(subclass)
            recurse(subclass)

    recurse(root_class)

    return set(subclass_list)


init_context_count = 0
top_level_context = None


def shutdown_init_context():
    """
    This function is used to initialize deepspeed engine inside the context of Init.
    We need to remove the wrappers but keep the context.
    """
    if top_level_context:
        top_level_context.unpatch_init_and_builtins()


def restore_init_context():
    """
    This function is used to restore the wrappers after deepspeed engine is initialized.
    """
    if top_level_context:
        top_level_context.patch_init_and_builtins()


# This context manager class is modified from
# class InsertPostInitMethodToModuleSubClasses in
# deepspeed/runtime/zero/partition_parameters.py
# 
# Inserts _post_init_method at the end of init method
# for the target module
class InsertPostInitMethodToModuleSubClasses(object):

    def __init__(self, enabled=True, dtype=None):
        self.enabled = enabled
        self.dtype = dtype or torch.half
        allowed_dtypes = [torch.half, torch.bfloat16, torch.float]
        assert self.dtype in allowed_dtypes, \
            f"Invalid data type {self.dtype}, allowed values are {allowed_dtypes}"

    def __enter__(self):
        if not self.enabled:
            return

        global init_context_count
        if init_context_count == 0:
            self.patch_init_and_builtins()
            global top_level_context
            top_level_context = self

        init_context_count += 1

    def __exit__(self, exc_type, exc_value, traceback):
        if not self.enabled:
            return

        global init_context_count
        init_context_count -= 1

        # Exiting the top level context
        if init_context_count == 0:
            self.unpatch_init_and_builtins()
            global top_level_context
            top_level_context = None

        # Now that we cleaned up the metaclass injection, raise the exception.
        if exc_type is not None:
            return False

    # To be implemented by inheriting classes
    def _post_init_method(self, module):
        pass

    def patch_init_and_builtins(self):

        def apply_with_gather(original_module_apply_fn: Callable) -> Callable:
            """many models make use of child modules like Linear or Embedding which
            perform their own weight initialization in their __init__ methods,
            but will then have more weight initialization in a parent module's __init__
            method that modifies weights of child modules, which is typically done
            using the Module.apply method.

            since the Init context manager partitions child modules immediately after
            they are initialized, without modifying apply we would entirely skip
            any initialization done by parent modules.

            to get around this issue, we wrap the function passed to Module.apply
            so that the applied function is applied to child modules correctly.
            """

            def get_wrapped_fn_to_apply(fn_to_apply: Callable) -> Callable:
                if hasattr(fn_to_apply, "wrapped"):
                    return fn_to_apply

                @functools.wraps(fn_to_apply)
                def wrapped_fn_to_apply(module_to_apply_fn_to: Module) -> None:
                    """gathers parameters before calling apply function. afterwards
                    parameters are broadcasted to ensure consistency across all ranks
                    then re-partitioned.

                    takes the following steps:
                    1. allgathers parameters for the current module being worked on
                    2. calls the original function
                    3. broadcasts root rank's parameters to the other ranks
                    4. re-partitions the parameters
                    """

                    # TODO Delay error checking for dangling partitioned parameters to post module init
                    # raise RuntimeError(f"not all parameters for {module_to_apply_fn_to.__class__.__name__}, "
                    #                    f"were zero params, is it possible that the parameters were "
                    #                    f"overwritten after they were initialized? "
                    #                    f"params: {[p for p in module_to_apply_fn_to.parameters(recurse=False)]} ")

                    params_to_apply_fn_to: Iterable[Parameter] = list(
                        sorted([p for p in module_to_apply_fn_to.parameters(recurse=False) if is_zero_param(p)],
                               key=lambda p: p.ds_id))

                    for param in params_to_apply_fn_to:
                        param.all_gather()

                    fn_to_apply(module_to_apply_fn_to)

                    for param in params_to_apply_fn_to:
                        dist.broadcast(param.data, 0, group=param.ds_process_group)

                    for param in params_to_apply_fn_to:
                        param.partition(has_been_updated=True)

                wrapped_fn_to_apply.wrapped = True

                return wrapped_fn_to_apply

            @functools.wraps(original_module_apply_fn)
            def wrapped_apply(module: Module, fn_to_apply: Callable) -> None:
                original_module_apply_fn(module, get_wrapped_fn_to_apply(fn_to_apply))

            return wrapped_apply

        def partition_after(f):

            @functools.wraps(f)
            def wrapper(module, *args, **kwargs):

                # important logic: We want to run post_init only after child's __init__ is
                # completed, and do nothing after __init__ of any of its parents and grandparents in
                # the inheritance ancestry. This way the partitioning will need to happen only once
                # when the whole object is ready to be partitioned and not before. This is because
                # often the child module will need to tweak the weights - for example running a
                # custom weights init function. So if a parent created the weights param, the child
                # won't need to gather it in order to tweak it

                is_child_module = False
                if not hasattr(module, "_ds_child_entered"):
                    # child's __init__ was called, since parents all see the same object they can now skip post_init
                    is_child_module = True
                    setattr(module, "_ds_child_entered", True)

                init_on_meta = 'device' in kwargs and kwargs['device'] == 'meta'
                if init_on_meta:
                    self.skip_init_depth += 1

                f(module, *args, **kwargs)
                if init_on_meta and self.skip_init_depth == 1:
                    # check and handle the logic of empty_init
                    hook_for_skip_init(module)
                if is_child_module:
                    # child's __init__ is done, now we can run a single post_init on the child object
                    delattr(module, "_ds_child_entered")

                    if self.skip_init_depth == 0:
                        self._post_init_method(module)

                if init_on_meta:
                    self.skip_init_depth -= 1

            return wrapper

        def _enable_class(_class):
            _class._original_init = _class.__init__
            _class.__init__ = partition_after(_class.__init__)

        def _init_subclass(_class, **kwargs):
            _class._original_init = _class.__init__
            _class.__init__ = partition_after(_class.__init__)

        # Replace .__init__() for all existing subclasses of torch.nn.Module recursively
        for subclass in get_all_subclasses(Module):
            _enable_class(subclass)

        # holding onto some methods so we can put them back the way they were in __exit__
        Module._original_init_subclass = Module.__init_subclass__
        Module._original_apply = Module.apply
        torch.Tensor.__original_new__ = torch.Tensor.__new__

        # Replace .__init__() for future subclasses of torch.nn.Module
        Module.__init_subclass__ = classmethod(_init_subclass)
        Module.apply = apply_with_gather(Module._original_apply)

        self._add_tensor_creation_wrappers()

        self.patched = True

    def unpatch_init_and_builtins(self):
        if self.patched:

            def _disable_class(_class):
                _class.__init__ = _class._original_init

            for subclass in get_all_subclasses(Module):
                _disable_class(subclass)

            # putting methods back the way we found them
            Module.__init_subclass__ = Module._original_init_subclass
            Module.apply = Module._original_apply

            self._remove_tensor_creation_wrappers()

            self.patched = False

    def _add_tensor_creation_wrappers(self):
        torch.Tensor.__new__ = get_new_tensor_fn_for_dtype(self.dtype)
        torch.tensor = fp_tensor_constructor_wrapper(_original_torch_tensor, self.dtype)
        torch.empty = fp_tensor_constructor_wrapper(_original_torch_empty, self.dtype)
        torch.zeros = fp_tensor_constructor_wrapper(_original_torch_zeros, self.dtype)
        torch.ones = fp_tensor_constructor_wrapper(_original_torch_ones, self.dtype)
        torch.full = fp_tensor_constructor_wrapper(_original_torch_full, self.dtype)
        torch.arange = fp_tensor_constructor_wrapper(_original_torch_arange, self.dtype)
        torch.eye = fp_tensor_constructor_wrapper(_original_torch_eye, self.dtype)
        torch.randn = fp_tensor_constructor_wrapper(_original_torch_randn, self.dtype)

    def _remove_tensor_creation_wrappers(self):
        torch.Tensor.__new__ = torch.Tensor.__original_new__
        torch.tensor = _original_torch_tensor
        torch.empty = _original_torch_empty
        torch.zeros = _original_torch_zeros
        torch.ones = _original_torch_ones
        torch.full = _original_torch_full
        torch.arange = _original_torch_arange
        torch.eye = _original_torch_eye
        torch.randn = _original_torch_randn


if __name__ == "__main__":
    with InsertPostInitMethodToModuleSubClasses():
        pass
