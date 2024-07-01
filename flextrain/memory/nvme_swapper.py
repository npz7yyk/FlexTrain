import os
import torch

from typing import Iterable

from flextrain.memory import (
    FlexTrainDataID,
    Waitable,
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
        aio_block_size: int,
        aio_queue_depth: int,
        aio_thread_count: int,
        aio_single_submit: bool,
        aio_overlap_events: bool
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
