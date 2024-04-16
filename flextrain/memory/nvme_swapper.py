import os
import torch

from typing import Iterable

from flextrain.memory import FlexTrainDataID
from flextrain.ops.aio import AsyncIOBuilder
from flextrain.defaults import (
    AIO_BLOCK_SIZE_DEFAULT,
    AIO_QUEUE_DEPTH_DEFAULT,
    AIO_THREAD_COUNT_DEFAULT,
    AIO_SINGLE_SUBMIT_DEFAULT,
    AIO_OVERLAP_EVENTS_DEFAULT
)

_AIO_DEFAULT_ARGS = [
    AIO_BLOCK_SIZE_DEFAULT,
    AIO_QUEUE_DEPTH_DEFAULT,
    AIO_SINGLE_SUBMIT_DEFAULT,
    AIO_OVERLAP_EVENTS_DEFAULT,
    AIO_THREAD_COUNT_DEFAULT
]


def swap_in_tensors(swap_handle, tensor_buffers, swap_paths):
    for buffer, path in zip(tensor_buffers, swap_paths):
        assert (swap_handle.async_pread(buffer, path) == 0)


def swap_out_tensors(swap_handle, tensor_buffers, swap_paths):
    for buffer, path in zip(tensor_buffers, swap_paths):
        assert (swap_handle.async_pwrite(buffer, path) == 0)


class Waitable:
    def wait(self):
        pass


class AsyncNVMeSwapper:

    def __init__(self, base_dir: str):
        # Directory to store the swapped tensors
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        # Ensure the directory is empty
        os.system(f"rm -rf {self.base_dir}/*")

        # Create the aio read and write handles
        aio_handle = AsyncIOBuilder().load().aio_handle
        self._aio_read_handle: Waitable = aio_handle(*_AIO_DEFAULT_ARGS)
        self._aio_write_handle: Waitable = aio_handle(*_AIO_DEFAULT_ARGS)

    def _filename(self, data_id: FlexTrainDataID):
        return os.path.join(self.base_dir, str(data_id))

    def swap_out(
        self,
        data_ids: FlexTrainDataID | Iterable[FlexTrainDataID],
        mem_srcs: torch.Tensor | Iterable[torch.Tensor],
        async_op: bool = False
    ):
        # If data_ids is not iterable, convert it to a list.
        if isinstance(data_ids, FlexTrainDataID):
            data_ids = [data_ids]
            mem_srcs = [mem_srcs]

        filenames = list(map(self._filename, data_ids))
        swap_out_tensors(self._aio_read_handle, mem_srcs, filenames)
        if async_op:
            return self._aio_read_handle
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
            return self._aio_write_handle
        else:
            assert self._aio_write_handle.wait() == len(filenames)
