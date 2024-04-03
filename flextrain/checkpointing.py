"""
FlexTrain checkpointing utils.

@NOTE:
FlexTrain checkpointing is not responsible for memory management.
It is the user's responsibility to ensure that:
1. Checkpointed activations are available for recomputation.
2. Offload activations to CPU or disk if necessary.
3. Input variables are detached by `detach_variable` between checkpoints.
"""

"""
Code for rng checkpointing taken from NVIDIA Megatron-LM mpu/random.py
b886b7bb972afe72bac0f5de4f42a4a7bae8ebef
"""

# Parts of the code here are adapted from PyTorch
# repo: https://github.com/pytorch/pytorch
import copy
import torch

from torch import _C

from deepspeed.accelerator import get_accelerator

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Tuple


def _set_cuda_rng_state(new_state):
    """
    Sets the random number generator state of the current GPU.

    Arguments:
        new_state (torch.ByteTensor): The desired state

    This function is adapted from PyTorch repo (torch.cuda.set_rng_state) #ignore-cuda
    with a single change: the input state is not cloned.
    Cloning caused major performance issues for +4 GPU cases.
    """
    if hasattr(_C, '_cuda_setRNGState') and callable(_C._cuda_setRNGState):
        # older PyTorch
        def cb():
            with get_accelerator().device(device):
                _C._cuda_setRNGState(new_state)
    else:
        # newer PyTorch
        device = torch.device(get_accelerator().device_name())

        def cb():
            idx = device.index
            if idx is None:
                idx = get_accelerator().current_device()
            default_generator = get_accelerator().default_generator(idx)
            default_generator.set_state(new_state)

    get_accelerator().lazy_call(cb)


class CudaRNGStatesTracker:
    """
    Tracker for the cuda RNG states.

    Using the `add` method, a cuda rng state is initialized based on
    the input `seed` and is assigned to `name`. Later, by forking the
    rng state, we can perform operations and return to our starting
    cuda state.
    """

    def __init__(self):
        # Map from a string name to the cuda rng state.
        self.states_ = {}
        # Seeds are just for book keeping and ensure no seed is set twice.
        self.seeds_ = set()

    def reset(self):
        """Set to the initial state (no tracker)."""
        self.states_ = {}
        self.seeds_ = set()

    def get_states(self):
        """
        Get rng states. Copy the dictionary so we have direct
        pointers to the states, not just a pointer to the dictionary.
        """
        return copy.copy(self.states_)

    def set_states(self, states):
        """
        Set the rng states. For efficiency purposes,
        we do not check the size of seed for compatibility.
        """
        self.states_ = states

    def add(self, name, seed):
        """Track the rng state."""
        # Check seed is not already used.
        if seed in self.seeds_:
            raise Exception('seed {} already exists'.format(seed))
        self.seeds_.add(seed)
        # Check that state is not already defined.
        if name in self.states_:
            raise Exception('cuda rng state {} already exists'.format(name))
        # Get the current rng state.
        orig_rng_state = get_accelerator().get_rng_state()
        # Set the new state and store it.
        get_accelerator().manual_seed(seed)
        self.states_[name] = get_accelerator().get_rng_state()
        # Reset rng state to what it was.
        _set_cuda_rng_state(orig_rng_state)


# RNG tracker object.
_CUDA_RNG_STATE_TRACKER = CudaRNGStatesTracker()


def data_parallel_cuda_manual_seed(seed):
    """
    Initialize data parallel cuda seed.

    This function should be called after the data parallel is initialized.
    No get_accelerator().manual_seed should be called after this function.
    Basically, this is replacement for that function.
    Two set of RNG states are tracked:
        default state: This used is for data parallelism and for example:
                       for dropout in the non-model-parallel regions.
    """

    # Data parallel gets the original seed.
    data_parallel_seed = seed

    _CUDA_RNG_STATE_TRACKER.reset()
    # Set the default state.
    get_accelerator().manual_seed(data_parallel_seed)


class RNGStatePack:
    """Pack the RNG states of CPU and CUDA."""

    def __init__(self, dummy=False):
        # If dummy, create a useless object to avoid conditionals.
        self.dummy = dummy
        if dummy: return

        # Get the RNG states.
        self.fwd_cpu_rng_state = torch.get_rng_state()
        self.fwd_cuda_rng_state = get_accelerator().get_rng_state()
        self.fwd_cuda_rng_state_tracker = _CUDA_RNG_STATE_TRACKER.get_states()

    def recover_states(self):
        """Recover the RNG states."""
        # If dummy, return immediately.
        if self.dummy: return

        torch.set_rng_state(self.fwd_cpu_rng_state)
        _set_cuda_rng_state(self.fwd_cuda_rng_state)
        _CUDA_RNG_STATE_TRACKER.set_states(self.fwd_cuda_rng_state_tracker)


@dataclass
class FWDContext:
    """Forward context for checkpointing."""

    # Function to run for the forward pass.
    run_function: Callable

    # RNG states before the forward pass.
    fwd_rng_state: RNGStatePack

    # Arguments to the forward pass.
    fwd_args: Tuple


def detach_variable(
    inputs: Iterable[Any | torch.Tensor] | Any | torch.Tensor
):
    """
    Detach the input variables and set requires_grad to True.
    """
    # If inputs is a single tensor, convert it to a list.
    unwrap_inputs = False
    if torch.is_tensor(inputs):
        inputs = [inputs]
        unwrap_inputs = True

    # If inputs is not iterable, return it as is.
    if not isinstance(inputs, Iterable):    
        return inputs

    # Detach the input variables and set requires_grad to True.
    # Assume that all tensor inputs require grad.
    rst = []
    for x in inputs:
        if torch.is_tensor(x):
            x = x.detach()
            x.requires_grad = True
        rst.append(x)

    return tuple(rst) if not unwrap_inputs else rst[0]


def checkpointed_forward(run_function, *args):
    """Conduct the forward and store necessary information for backward.

    Args:
        run_function (Callable): The forward function to run.
        *args: The arguments to the forward function.

    Returns:
        Tuple(
            The outputs of the forward function,
            FWDContext: The context used for recomputation
        )
    """

    # Copy the rng states before the forward pass.
    fwd_rng_state = RNGStatePack()

    # Run the forward pass.
    with torch.no_grad():
        outputs = run_function(*args)

    # Store necessary information for backward pass.
    fwd_ctx = FWDContext(
        run_function=run_function,
        fwd_rng_state=fwd_rng_state,
        fwd_args=args
    )

    return outputs, fwd_ctx


def retrieve_tensor_grads(
    tensors: Iterable[Any | torch.Tensor] | Any | torch.Tensor
):
    """
    Retrieve the gradients of the input tensors.
    """
    # If tensors is a single tensor, convert it to a list.
    unwrap_tensors = False
    if torch.is_tensor(tensors):
        tensors = [tensors]
        unwrap_tensors = True

    # If tensors is not iterable, return it as is.
    if not isinstance(tensors, Iterable):    
        return tensors

    # Retrieve the gradients of the input tensors.
    rst = []
    for x in tensors:
        if torch.is_tensor(x):
            rst.append(x.grad)
        else:
            rst.append(None)

    return tuple(rst) if not unwrap_tensors else rst[0]


def checkpointed_backward(fwd_ctx: FWDContext, *grads):
    """Conduct the recomputation and backward pass.

    Args:
        fwd_ctx (FWDContext): The forward context.
        *grads: The gradients to backpropagate.

    Returns:
        Tuple: The gradients for the inputs.
    """

    # Store the current states.
    bwd_rng_state = RNGStatePack()

    # Recover the forward RNG states and inputs.
    inputs = fwd_ctx.fwd_args
    fwd_ctx.fwd_rng_state.recover_states()

    # Run the forward pass again to get the intermediate tensors.
    with torch.enable_grad():
        outputs = fwd_ctx.run_function(*inputs)

    # Recover the RNG states.
    bwd_rng_state.recover_states()

    if isinstance(outputs, torch.Tensor):
        outputs = (outputs, )

    # Filter out non tensor outputs
    outputs = [out for out in outputs if torch.is_tensor(out)]

    # Construct arguments to autograd.backward()
    output_tensors = []
    grad_tensors = []
    for out, grad in zip(outputs, grads):
        if out.requires_grad:
            output_tensors.append(out)
            grad_tensors.append(grad)

    # Run the backward pass.
    torch.autograd.backward(output_tensors, grad_tensors)

    # Force clear forward context to prevent a memory leak in certain scenarios
    del fwd_ctx

    # Return the gradients for the inputs
    grad_list = []
    for x in inputs:
        if torch.is_tensor(x):
            grad_list.append(x.grad)
        else:
            grad_list.append(None)

    return tuple(grad_list) if len(grad_list) > 1 else grad_list[0]
