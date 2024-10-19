import torch

from dataclasses import dataclass
from torch import Tensor
from typing import Tuple

from flextrain.config import get_flextrain_config
from flextrain.memory import (
    RotateContainer,
    Waitable,
    DummyHandle,
    FunctionHandle,
    get_split_numels,
    allocate_memory_chunks,
    free_tensor,
    get_data_stream
)
from flextrain.memory.coordinator import get_para_coordinator
from flextrain.utils import dist
from flextrain.utils.logging import rank0_logger


@dataclass
class InterLayerTask:
    unit: int
    micro_batch: int
    tensor: Tensor = None


def retrieve_tensor(interlayer: Tensor | Tuple[Tensor, ...]) -> Tensor:
    # If interlayer is a single tensor, return it.
    if isinstance(interlayer, Tensor):
        return interlayer

    # Unpack the tuple.
    # Currently, only one tensor is supported.
    tar = None
    for tensor in interlayer:
        if isinstance(tensor, Tensor):
            assert tar is None, (
                "Currently, only one tensor is supported for FlexTrain "
                "checkpointing. You may consider manually place all "
                "inter-layer results into a single tensor."
            )
            tar = tensor

    assert tar is not None, "No tensor can be found in inter-layer results."
    return tar


class FlexTrainInterLayerCoordinator:

    def __init__(self):
        # Lazy initialization of checkpoint coordinator.
        self._initialized = False

    @property
    def is_initialized(self):
        return self._initialized

    def _init_coordinator(self, tensor: Tensor):
        # If already initialized, return.
        if self._initialized:
            return
        self._initialized = True

        # Memory lazy allocation.
        self._mem_allocated = False

        # Record the tensor shape.
        self._tensor_shape = tensor.shape

        # CUDA streams for async IO operations.
        self._data_stream = get_data_stream()
        # Last task handle.
        self._inflight_handle: Waitable = DummyHandle()

        # Initialize coordinator configurations.
        config = get_flextrain_config()
        self._num_units = get_para_coordinator().num_units
        num_micro_batches = config.batch_size // config.micro_batch_size
        self._micro_batch_per_rank = num_micro_batches // dist.get_world_size()

        # How to split the checkpoint tensor.
        self._ckpt_numels = get_split_numels(
            tensor.numel(), config.split_ratio.checkpoint, num_levels=2
        )
        # How to split the gradient tensor.
        self._grad_numels = get_split_numels(
            tensor.numel(), config.split_ratio.gradient, num_levels=2
        )

        # Allocate memory for checkpoint.
        self.gpu_ckpt_base = allocate_memory_chunks(
            self._ckpt_numels[0],
            (self._num_units, self._micro_batch_per_rank),
            config.mixed_precision.device_dtype,
            torch.cuda.current_device()
        )
        self.cpu_ckpt_base = allocate_memory_chunks(
            self._ckpt_numels[1],
            (self._num_units, self._micro_batch_per_rank),
            config.mixed_precision.device_dtype,
            torch.device('cpu')
        )

        # Allocate memory for gradient.
        self.gpu_grad_base = allocate_memory_chunks(
            self._grad_numels[0],
            self._micro_batch_per_rank,
            config.mixed_precision.device_dtype,
            torch.cuda.current_device()
        )
        self.cpu_grad_base = allocate_memory_chunks(
            self._grad_numels[1],
            self._micro_batch_per_rank,
            config.mixed_precision.device_dtype,
            torch.device('cpu')
        )

        # Allocate memory for GPU checkpoint buffer.
        self._gpu_full_ckpts = RotateContainer(
            allocate_memory_chunks(
                tensor.numel(), 2,
                config.mixed_precision.device_dtype,
                torch.cuda.current_device()
            )
        )
        # Allocate memory for GPU gradient buffer.
        self._gpu_full_grads = RotateContainer(
            allocate_memory_chunks(
                tensor.numel(), 2,
                config.mixed_precision.device_dtype,
                torch.cuda.current_device()
            )
        )

        # Log the configuration.
        rank0_logger.info(
            "\n\n> "
            f"FlexTrain inter-layer coordinator initialized "
            f"with configurations:\n"
            f"  - checkpoint split numels (GPU, CPU): {self._ckpt_numels}\n"
            f"  - gradient split numels (GPU, CPU): {self._grad_numels}\n"
        )

    def _mask_invalid_task(self, task: InterLayerTask):
        if not self._initialized or task is None:
            return task
        unit = task.unit
        micro_batch = task.micro_batch
        if unit < 0 or unit >= self._num_units - 1:
            return None
        elif micro_batch < 0 or micro_batch >= self._micro_batch_per_rank:
            return None
        else:
            return task

    @property
    def inflight_layer_ckpt(self):
        return self._gpu_full_ckpts[0]

    @property
    def available_layer_ckpt(self):
        ckpt_mem: Tensor = self._gpu_full_ckpts[1]
        return ckpt_mem.view(self._tensor_shape)

    @property
    def inflight_layer_grad(self):
        return self._gpu_full_grads[0]

    @property
    def available_layer_grad(self):
        grad_mem: Tensor = self._gpu_full_grads[1]
        return grad_mem.view(self._tensor_shape)

    def _sync_pre_micro_batch_forward(self):
        self._inflight_handle.wait()
        self._gpu_full_ckpts.rotate()

    def _prefetch_ckpt(self, task: InterLayerTask):
        tar = self.inflight_layer_ckpt
        gpu_tar, cpu_tar = tar.split(self._ckpt_numels)
        gpu_src = self.gpu_ckpt_base[task.unit][task.micro_batch]
        cpu_src = self.cpu_ckpt_base[task.unit][task.micro_batch]
        gpu_tar.copy_(gpu_src, non_blocking=True)
        cpu_tar.copy_(cpu_src, non_blocking=True)

    def _offload_ckpt(self, task: InterLayerTask):
        src = task.tensor.flatten()
        gpu_src, cpu_src = src.split(self._ckpt_numels)
        gpu_tar = self.gpu_ckpt_base[task.unit][task.micro_batch]
        cpu_tar = self.cpu_ckpt_base[task.unit][task.micro_batch]
        gpu_tar.copy_(gpu_src, non_blocking=True)
        cpu_tar.copy_(cpu_src, non_blocking=True)

    def _submit_pre_micro_batch_forward(
        self,
        ckpt_prefetch: InterLayerTask = None,
        ckpt_offload: InterLayerTask = None
    ):
        def interlayer_task():
            if ckpt_prefetch is not None:
                self._prefetch_ckpt(ckpt_prefetch)
            if ckpt_offload is not None:
                self._offload_ckpt(ckpt_offload)

        self._data_stream.submit(interlayer_task)

        def free_ckpt_memory():
            if ckpt_offload is not None and ckpt_offload.tensor is not None:
                free_tensor(ckpt_offload.tensor)

        self._inflight_handle = FunctionHandle(free_ckpt_memory)

    def pre_micro_batch_forward(
        self,
        ckpt_prefetch: InterLayerTask = None,
        ckpt_offload: InterLayerTask = None
    ):
        # Mask invalid tasks.
        ckpt_prefetch = self._mask_invalid_task(ckpt_prefetch)
        ckpt_offload = self._mask_invalid_task(ckpt_offload)

        # Initialize coordinator if not done.
        if not self._initialized:
            if ckpt_offload is not None:
                self._init_coordinator(ckpt_offload.tensor)

        # If still not initialized, return.
        if not self._initialized:
            return

        self._sync_pre_micro_batch_forward()
        self._submit_pre_micro_batch_forward(ckpt_prefetch, ckpt_offload)

    def _sync_pre_micro_batch_backward(self):
        self._inflight_handle.wait()
        self._gpu_full_ckpts.rotate()
        self._gpu_full_grads.rotate()

    def _grad_prefetch(self, task: InterLayerTask):
        tar = self.inflight_layer_grad
        gpu_tar, cpu_tar = tar.split(self._grad_numels)
        gpu_src = self.gpu_grad_base[task.micro_batch]
        cpu_src = self.cpu_grad_base[task.micro_batch]
        gpu_tar.copy_(gpu_src, non_blocking=True)
        cpu_tar.copy_(cpu_src, non_blocking=True)

    def _grad_offload(self, task: InterLayerTask):
        src = task.tensor.flatten()
        gpu_src, cpu_src = src.split(self._grad_numels)
        gpu_tar = self.gpu_grad_base[task.micro_batch]
        cpu_tar = self.cpu_grad_base[task.micro_batch]
        gpu_tar.copy_(gpu_src, non_blocking=True)
        cpu_tar.copy_(cpu_src, non_blocking=True)

    def _submit_pre_micro_batch_backward(
        self,
        ckpt_prefetch: InterLayerTask = None,
        grad_prefetch: InterLayerTask = None,
        grad_offload: InterLayerTask = None
    ):
        def interlayer_task():
            if ckpt_prefetch is not None:
                self._prefetch_ckpt(ckpt_prefetch)
            if grad_prefetch is not None:
                self._grad_prefetch(grad_prefetch)
            if grad_offload is not None:
                self._grad_offload(grad_offload)

        self._data_stream.submit(interlayer_task)

        def free_grad_memory():
            if grad_offload is not None and grad_offload.tensor is not None:
                free_tensor(grad_offload.tensor)

        self._inflight_handle = FunctionHandle(free_grad_memory)

    def pre_micro_batch_backward(
        self,
        ckpt_prefetch: InterLayerTask = None,
        grad_prefetch: InterLayerTask = None,
        grad_offload: InterLayerTask = None
    ):
        # Mask invalid tasks.
        ckpt_prefetch = self._mask_invalid_task(ckpt_prefetch)
        grad_prefetch = self._mask_invalid_task(grad_prefetch)
        grad_offload = self._mask_invalid_task(grad_offload)

        self._sync_pre_micro_batch_backward()
        self._submit_pre_micro_batch_backward(
            ckpt_prefetch, grad_prefetch, grad_offload
        )


_INTERLAYER_COORDINATOR = FlexTrainInterLayerCoordinator()


def get_interlayer_coordinator():
    """
    Get the inter-layer coordinator.

    Returns:
        FlexTrainInterLayerCoordinator: The inter-layer coordinator.
    """
    return _INTERLAYER_COORDINATOR
