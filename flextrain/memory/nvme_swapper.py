import os
import torch

from torch import Tensor
from typing import Iterable, Tuple

from flextrain.config import get_flextrain_config
from flextrain.defaults import (
    AIO_BLOCK_SIZE_DEFAULT,
    AIO_QUEUE_DEPTH_DEFAULT,
    AIO_THREAD_COUNT_DEFAULT,
    AIO_SINGLE_SUBMIT_DEFAULT,
    AIO_OVERLAP_EVENTS_DEFAULT
)
from flextrain.memory import (
    FlexTrainDataID,
    Waitable,
    DummyHandle,
    AsyncIOHandle
)
from flextrain.ops.aio import AsyncIOBuilder


def swap_in_tensors(swap_handle, tensor_buffers, swap_paths):
    for buffer, path in zip(tensor_buffers, swap_paths):
        assert (swap_handle.async_pread(buffer, path) == 0)


def swap_out_tensors(swap_handle, tensor_buffers, swap_paths):
    for buffer, path in zip(tensor_buffers, swap_paths):
        assert (swap_handle.async_pwrite(buffer, path) == 0)


class AsyncNVMeSwapper:

    def __init__(
        self,
        swap_dir: str,
        aio_block_size: int = AIO_BLOCK_SIZE_DEFAULT,
        aio_queue_depth: int = AIO_QUEUE_DEPTH_DEFAULT,
        aio_thread_count: int = AIO_THREAD_COUNT_DEFAULT,
        aio_single_submit: bool = AIO_SINGLE_SUBMIT_DEFAULT,
        aio_overlap_events: bool = AIO_OVERLAP_EVENTS_DEFAULT
    ):
        # Directory to store the swapped tensors
        self.swap_dir = swap_dir
        os.makedirs(self.swap_dir, exist_ok=True)
        # Ensure the directory is empty
        os.system(f"rm -rf {self.swap_dir}/*")

        # Create the aio read and write handles
        aio_handle = AsyncIOBuilder().load().aio_handle
        self._aio_read_handle: Waitable = aio_handle(
            aio_block_size,
            aio_queue_depth,
            aio_thread_count,
            aio_single_submit,
            aio_overlap_events
        )
        self._aio_write_handle: Waitable = aio_handle(
            aio_block_size,
            aio_queue_depth,
            aio_thread_count,
            aio_single_submit,
            aio_overlap_events
        )

    def _filename(self, data_id: FlexTrainDataID):
        return os.path.join(self.swap_dir, str(data_id))

    def swap_out(
        self,
        data_ids: FlexTrainDataID | Iterable[FlexTrainDataID],
        mem_srcs: torch.Tensor | Iterable[torch.Tensor],
        async_op: bool = False
    ):
        # If data_ids is not iterable, convert it to a list.
        if isinstance(data_ids, FlexTrainDataID):
            data_ids = [data_ids]
        if isinstance(data_ids, str):
            data_ids = [data_ids]
        if isinstance(mem_srcs, torch.Tensor):
            mem_srcs = [mem_srcs]

        filenames = list(map(self._filename, data_ids))
        swap_out_tensors(self._aio_read_handle, mem_srcs, filenames)
        if async_op:
            return AsyncIOHandle(self._aio_read_handle) \
                if async_op else None
        else:
            assert self._aio_read_handle.wait() == len(filenames)

    def swap_in(
        self,
        data_ids: FlexTrainDataID | Iterable[FlexTrainDataID],
        mem_dsts: torch.Tensor | Iterable[torch.Tensor],
        async_op: bool = False
    ):
        # If data_ids is not iterable, convert it to a list.
        if isinstance(data_ids, FlexTrainDataID):
            data_ids = [data_ids]
            mem_dsts = [mem_dsts]

        filenames = list(map(self._filename, data_ids))
        swap_in_tensors(self._aio_write_handle, mem_dsts, filenames)
        if async_op:
            return AsyncIOHandle(self._aio_write_handle) \
                if async_op else None
        else:
            assert self._aio_write_handle.wait() == len(filenames)


_NVME_SWAPPER: AsyncNVMeSwapper = None


def _filter_tensors(
    data_ids: FlexTrainDataID | Iterable[FlexTrainDataID],
    tensors: Tensor | Iterable[Tensor]
):
    if isinstance(data_ids, FlexTrainDataID):
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


def _nvme_offload(
    data_ids: FlexTrainDataID | Iterable[FlexTrainDataID],
    tensors: Tensor | Iterable[Tensor],
    async_op: bool = False
):
    # Filter out empty tensors.
    data_ids, tensors = _filter_tensors(data_ids, tensors)

    # If no tensors to offload, return.
    if len(data_ids) == 0:
        return DummyHandle() if async_op else None

    # Lazy initialization of the NVMe swapper.
    global _NVME_SWAPPER
    if _NVME_SWAPPER is None:
        nvme_swap_config = get_flextrain_config().nvme_swap
        _NVME_SWAPPER = AsyncNVMeSwapper(
            swap_dir=nvme_swap_config.swap_dir,
            aio_block_size=nvme_swap_config.aio_block_size,
            aio_queue_depth=nvme_swap_config.aio_queue_depth,
            aio_thread_count=nvme_swap_config.aio_thread_count,
            aio_single_submit=nvme_swap_config.aio_single_submit,
            aio_overlap_events=nvme_swap_config.aio_overlap_events
        )

    # Call the NVMe swapper.
    return _NVME_SWAPPER.swap_out(data_ids, tensors, async_op)


def _nvme_reload(
    data_ids: FlexTrainDataID | Iterable[FlexTrainDataID],
    tensors: Tensor | Iterable[Tensor],
    async_op: bool = False
):
    # If swapper is not initialized, return.
    if _NVME_SWAPPER is None:
        return DummyHandle() if async_op else None

    # Filter out empty tensors.
    data_ids, tensors = _filter_tensors(data_ids, tensors)

    # If no tensors to reload, return.
    if len(data_ids) == 0:
        return DummyHandle() if async_op else None

    # Call the NVMe swapper.
    return _NVME_SWAPPER.swap_in(data_ids, tensors, async_op)


class NVMeGroup:

    def __init__(self, numels: Tuple[int, ...]):
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
    ) -> Waitable:
        assert tensor.numel() == self._numels[index]
        return _nvme_offload(self._rename(prefix, index), tensor, async_op)

    def group_offload(
        self,
        prefix: FlexTrainDataID,
        tensors: Tensor | Iterable[Tensor],
        async_op: bool = False
    ) -> Waitable:
        if isinstance(tensors, Tensor):
            assert tensors.numel() == self._group_numel, (
                f"Expected numel={self._group_numel}, "
                f"got numel={tensors.numel()}"
            )
            tensors = torch.split(tensors, self._numels)

        for tensor, numel in zip(tensors, self._numels):
            assert tensor.numel() == numel

        return _nvme_offload(
            [self._rename(prefix, i) for i in range(len(tensors))],
            tensors, async_op
        )

    def single_reload(
        self,
        prefix: FlexTrainDataID,
        tensor: Tensor,
        index: int,
        async_op: bool = False
    ) -> Waitable:
        assert tensor.numel() == self._numels[index]
        return _nvme_reload(self._rename(prefix, index), tensor, async_op)

    def group_reload(
        self,
        prefix: FlexTrainDataID,
        tensors: Tensor | Iterable[Tensor],
        async_op: bool = False
    ) -> Waitable:
        if isinstance(tensors, Tensor):
            assert tensors.numel() == self._group_numel
            tensors = torch.split(tensors, self._numels)

        for tensor, numel in zip(tensors, self._numels):
            assert tensor.numel() == numel

        return _nvme_reload(
            [self._rename(prefix, i) for i in range(len(tensors))],
            tensors, async_op
        )
