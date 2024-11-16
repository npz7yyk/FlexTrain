import os
import torch

from torch import Tensor
from typing import Iterable, Tuple

from flextrain.config import get_flextrain_config
from flextrain.memory import FlexTrainDataID, Waitable
from flextrain.ops.aio import AsyncIOBuilder


def swap_in_tensors(swap_handle, tensor_buffers, swap_paths):
    for buffer, path in zip(tensor_buffers, swap_paths):
        assert (swap_handle.async_pread(buffer, path) == 0)


def swap_out_tensors(swap_handle, tensor_buffers, swap_paths):
    for buffer, path in zip(tensor_buffers, swap_paths):
        assert (swap_handle.async_pwrite(buffer, path) == 0)


def _filter_tensors(
    data_ids: FlexTrainDataID | Iterable[FlexTrainDataID],
    tensors: Tensor | Iterable[Tensor]
):
    if isinstance(data_ids, (FlexTrainDataID, str)):
        data_ids = [data_ids]
    if isinstance(tensors, Tensor):
        tensors = [tensors]

    non_empty_data_ids = []
    non_empty_tensors = []
    for data_id, tensor in zip(data_ids, tensors):
        if tensor.numel() == 0:
            continue
        non_empty_data_ids.append(data_id)
        non_empty_tensors.append(tensor)

    return non_empty_data_ids, non_empty_tensors


class AsyncNVMeSwapper:

    def __init__(self):
        self._initialized = False

    def _init_swapper(self):
        # If already initialized, return.
        if self._initialized:
            return
        self._initialized = True

        # Config for NVMe swapping.
        nvme_swap_config = get_flextrain_config().nvme_swap

        # Directory to store the swapped tensors.
        self.swap_dir = nvme_swap_config.swap_dir
        os.makedirs(self.swap_dir, exist_ok=True)
        # Ensure the directory is empty.
        os.system(f"rm -rf {self.swap_dir}/*")

        # Create the aio read and write handles.
        aio_handle = AsyncIOBuilder().load().aio_handle
        self._aio_read_handle: Waitable = aio_handle(
            nvme_swap_config.aio_block_size,
            nvme_swap_config.aio_queue_depth,
            nvme_swap_config.aio_thread_count,
            nvme_swap_config.aio_single_submit,
            nvme_swap_config.aio_overlap_events
        )
        self._aio_write_handle: Waitable = aio_handle(
            nvme_swap_config.aio_block_size,
            nvme_swap_config.aio_queue_depth,
            nvme_swap_config.aio_thread_count,
            nvme_swap_config.aio_single_submit,
            nvme_swap_config.aio_overlap_events
        )

        # Numbers of inflight read and write handles.
        self._inflight_read_tasks = 0
        self._inflight_write_tasks = 0

    def is_empty(self):
        # If not initialized, return True.
        if not self._initialized:
            return True
        # Otherwise, return if there are no inflight read or write tasks.
        return not self._inflight_read_tasks and \
            not self._inflight_write_tasks

    def _filename(self, data_id: FlexTrainDataID):
        return os.path.join(self.swap_dir, str(data_id))

    def swap_out(
        self,
        data_ids: FlexTrainDataID | Iterable[FlexTrainDataID],
        mem_srcs: torch.Tensor | Iterable[torch.Tensor],
        async_op: bool = False
    ):
        # Filter out empty tensors.
        data_ids, mem_srcs = _filter_tensors(data_ids, mem_srcs)

        # If no tensors to offload, return.
        if len(data_ids) == 0:
            return

        # Initialize the swapper.
        self._init_swapper()

        filenames = list(map(self._filename, data_ids))
        swap_out_tensors(self._aio_write_handle, mem_srcs, filenames)
        if async_op:
            self._inflight_write_tasks += len(filenames)
        else:
            assert self._aio_write_handle.wait() == len(filenames)

    def swap_in(
        self,
        data_ids: FlexTrainDataID | Iterable[FlexTrainDataID],
        mem_dsts: torch.Tensor | Iterable[torch.Tensor],
        async_op: bool = False
    ):
        # Filter out empty tensors.
        data_ids, mem_dsts = _filter_tensors(data_ids, mem_dsts)

        # If no tensors to reload, return.
        if len(data_ids) == 0:
            return

        filenames = list(map(self._filename, data_ids))
        swap_in_tensors(self._aio_read_handle, mem_dsts, filenames)
        if async_op:
            self._inflight_read_tasks += len(filenames)
        else:
            assert self._aio_read_handle.wait() == len(filenames)

    def synchronize(self):
        # If not initialized, return.
        if not self._initialized:
            return
        # Synchronize inflight read tasks if needed.
        if self._inflight_read_tasks:
            self._aio_read_handle.wait() == self._inflight_read_tasks
            self._inflight_read_tasks = 0
        # Synchronize inflight write tasks if needed.
        if self._inflight_write_tasks:
            self._aio_write_handle.wait() == self._inflight_write_tasks
            self._inflight_write_tasks = 0


_NVME_SWAPPER = AsyncNVMeSwapper()


def get_nvme_swapper():
    """
    Get the async IO NVMe swapper.

    Returns:
        AsyncNVMeSwapper: The async IO NVMe swapper.
    """
    return _NVME_SWAPPER


class NVMeGroup:

    def __init__(self, numels: Tuple[int, ...]):
        self._nvme_swapper = get_nvme_swapper()
        self._numels = numels
        self._group_numel = sum(numels)

    def _rename(self, prefix: FlexTrainDataID, index: int):
        return str(prefix) + f".{index}"

    def single_offload(
        self,
        prefix: FlexTrainDataID,
        tensor: Tensor,
        index: int,
        async_op: bool = False
    ):
        assert tensor.numel() == self._numels[index]
        self._nvme_swapper.swap_out(
            self._rename(prefix, index), tensor, async_op
        )

    def group_offload(
        self,
        prefix: FlexTrainDataID,
        tensors: Tensor | Iterable[Tensor],
        async_op: bool = False
    ):
        if isinstance(tensors, Tensor):
            assert tensors.numel() == self._group_numel, (
                f"Expected numel={self._group_numel}, "
                f"got numel={tensors.numel()}"
            )
            tensors = torch.split(tensors, self._numels)

        for tensor, numel in zip(tensors, self._numels):
            assert tensor.numel() == numel

        self._nvme_swapper.swap_out(
            [self._rename(prefix, i) for i in range(len(tensors))],
            tensors, async_op
        )

    def single_reload(
        self,
        prefix: FlexTrainDataID,
        tensor: Tensor,
        index: int,
        async_op: bool = False
    ):
        assert tensor.numel() == self._numels[index]
        self._nvme_swapper.swap_in(
            self._rename(prefix, index), tensor, async_op
        )

    def group_reload(
        self,
        prefix: FlexTrainDataID,
        tensors: Tensor | Iterable[Tensor],
        async_op: bool = False
    ):
        if isinstance(tensors, Tensor):
            assert tensors.numel() == self._group_numel
            tensors = torch.split(tensors, self._numels)

        for tensor, numel in zip(tensors, self._numels):
            assert tensor.numel() == numel

        self._nvme_swapper.swap_in(
            [self._rename(prefix, i) for i in range(len(tensors))],
            tensors, async_op
        )
