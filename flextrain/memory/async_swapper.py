import os
import torch

from flextrain.memory.data_id import FlexTrainDataID
from flextrain.ops.aio import AsyncIOBuilder
from flextrain.defaults import (
    AIO_BLOCK_SIZE_DEFAULT,
    AIO_QUEUE_DEPTH_DEFAULT,
    AIO_THREAD_COUNT_DEFAULT,
    AIO_SINGLE_SUBMIT_DEFAULT,
    AIO_OVERLAP_EVENTS_DEFAULT
)


AIO_BLOCK_SIZE = "block_size"
AIO_QUEUE_DEPTH = "queue_depth"
AIO_THREAD_COUNT = "thread_count"
AIO_SINGLE_SUBMIT = "single_submit"
AIO_OVERLAP_EVENTS = "overlap_events"

AIO_DEFAULT_DICT = {
    AIO_BLOCK_SIZE: AIO_BLOCK_SIZE_DEFAULT,
    AIO_QUEUE_DEPTH: AIO_QUEUE_DEPTH_DEFAULT,
    AIO_THREAD_COUNT: AIO_THREAD_COUNT_DEFAULT,
    AIO_SINGLE_SUBMIT: AIO_SINGLE_SUBMIT_DEFAULT,
    AIO_OVERLAP_EVENTS: AIO_OVERLAP_EVENTS_DEFAULT
}


def swap_in_tensors(swap_handle, tensor_buffers, swap_paths):
    for buffer, path in zip(tensor_buffers, swap_paths):
        assert (swap_handle.async_pread(buffer, path) == 0)


def swap_out_tensors(swap_handle, tensor_buffers, swap_paths):
    for buffer, path in zip(tensor_buffers, swap_paths):
        assert (swap_handle.async_pwrite(buffer, path) == 0)


class AsyncSwapper:

    def __init__(self, base_dir: str):
        # Directory to store the swapped tensors
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

        # Create the aio read and write handles
        aio_handle = AsyncIOBuilder().load().aio_handle
        self._aio_read_handle = aio_handle(**AIO_DEFAULT_DICT)
        self._aio_write_handle = aio_handle(**AIO_DEFAULT_DICT)

    def _filename(self, data_id: FlexTrainDataID):
        return os.path.join(self.base_dir, str(data_id))

    def allocated_storage(
        self,
        data_id: FlexTrainDataID,
        size_in_bytes: int,
        temp_cpu_buffer: torch.Tensor = None
    ):
        try:
            filename = self._filename(data_id)
            if temp_cpu_buffer is None:
                temp_cpu_buffer = torch.empty(size_in_bytes, dtype=torch.uint8)
            swap_in_tensors(self._aio_read_handle, [temp_cpu_buffer], [filename])
        except BaseException:
            raise RuntimeError(
                f"Failed to allocate dist storage for {data_id}, "
                f"which requires {size_in_bytes} bytes."
            )
