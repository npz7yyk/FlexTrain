from dataclasses import dataclass
from typing import Tuple


@dataclass
class LLMTask:
    """ FlexTrain LLM Task.

    Arguments:
        is_forwarding (bool): True if the task is forwarding, else False
        micro_batch (int): Micro batch index in the batch
        layers (Tuple[int]): Tuple of layers to be executed
    """
    is_forwarding: bool
    micro_batch: int
    layers: Tuple[int]


class GreedySnakeBlockScheduler:
    """ FlexTrain Greedy Snake Block Scheduler.

    Arguments:
        micro_batch_per_block (int): Micro batch per block
        num_layers (int): Number of layers in the given LLM
        checkpoint_interval (int): Checkpoint interval
    """

    def __init__(
        self,
        micro_batch_per_block,
        num_layers,
        checkpoint_interval
    ):
        self.micro_batch_per_block = micro_batch_per_block
        self.num_layers = num_layers
        self.checkpoint_interval = checkpoint_interval

    def __iter__(self):
        curr_layer = 0
        top_down = True

        # forwarding
        while True:
            head_layer = curr_layer
            tail_layer = curr_layer + self.checkpoint_interval

            # working window reaches the end, convert to backwarding
            if tail_layer >= self.num_layers:
                break

            curr_layers = tuple(range(head_layer, tail_layer))
            for micro_step in range(self.micro_batch_per_block):
                curr_micro_batch = micro_step if top_down \
                    else self.micro_batch_per_block - 1 - micro_step
                yield LLMTask(True, curr_micro_batch, curr_layers)

            # reverse direction
            top_down = not top_down

            # move to next checkpoint
            curr_layer = tail_layer

        # backwarding
        while True:
            head_layer = curr_layer
            tail_layer = curr_layer + self.checkpoint_interval
            tail_layer = min(tail_layer, self.num_layers)

            # working window reaches the start, end the loop
            if head_layer < 0:
                break

            curr_layers = tuple(range(head_layer, tail_layer))
            for micro_step in range(self.micro_batch_per_block):
                curr_micro_batch = micro_step if top_down \
                    else self.micro_batch_per_block - 1 - micro_step
                yield LLMTask(False, curr_micro_batch, curr_layers)

            # reverse direction
            top_down = not top_down

            # move to next checkpoint
            curr_layer = head_layer - self.checkpoint_interval


class GreedySnakeBatchScheduler:
    """ FlexTrain Greedy Snake Batch Scheduler.

    Arguments:
        world_size (int): World size
        batch_size (int): Training batch size provided by data scientist
        micro_batch_size (int): Micro batch size
        micro_batch_per_block (int): Micro batch per block
        num_layers (int): Number of layers in the given LLM
        checkpoint_interval (int): Checkpoint interval
    """

    def __init__(
        self,
        world_size,
        batch_size,
        micro_batch_size,
        micro_batch_per_block,
        num_layers,
        checkpoint_interval
    ):
        assert batch_size % world_size == 0, \
            "Batch size must be divisible by world size"

        sample_per_device = batch_size // world_size

        assert sample_per_device % micro_batch_size == 0, \
            "Sample per device must be divisible by micro batch size"

        micro_batch_per_device = sample_per_device // micro_batch_size
        assert micro_batch_per_device >= micro_batch_per_block, \
            "Block must be smaller than batch"

        self.micro_batch_per_device = micro_batch_per_device
        self.micro_batch_per_block = micro_batch_per_block
        self.num_layers = num_layers
        self.checkpoint_interval = checkpoint_interval

        micro_batch_last_block = \
            (micro_batch_per_device - 1) % micro_batch_per_block + 1
        self.micro_batch_last_block = micro_batch_last_block
        self.norm_block_scheduler = GreedySnakeBlockScheduler(
            micro_batch_per_block, num_layers, checkpoint_interval
        )
        self.last_block_scheduler = GreedySnakeBlockScheduler(
            micro_batch_last_block, num_layers, checkpoint_interval
        )

    def __iter__(self):
        curr_block_micro_batch_base = 0
        last_block_micro_batch_base = \
            self.micro_batch_per_device - self.micro_batch_last_block

        while curr_block_micro_batch_base < last_block_micro_batch_base:
            for task in self.norm_block_scheduler:
                task.micro_batch += curr_block_micro_batch_base
                yield task
            curr_block_micro_batch_base += self.micro_batch_per_block

        for task in self.last_block_scheduler:
            task.micro_batch += curr_block_micro_batch_base
            yield task

    @property
    def current_scheduler(self):
        ...

    @property
    def activation_reusable(self):
        ...

    @property
    def at_optimizer_step_boundary(self):
        ...

    @property
    def at_window_moving_boundary(self):
        ...
