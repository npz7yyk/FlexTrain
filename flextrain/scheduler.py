from dataclasses import dataclass


@dataclass
class LLMTask:
    """ FlexTrain LLM Task.

    Arguments:
        is_forwarding (bool): True if the task is forwarding, else False
        micro_batch (int): Micro batch index in the batch
        unit (int): Unit index in the LLM to be executed
    """
    is_forwarding: bool
    micro_batch: int
    unit: int


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
        num_units
    ):
        self.micro_batch_per_block = micro_batch_per_block
        self.num_units = num_units

    def __iter__(self):
        curr_unit = 0
        self.top_down = True

        # Keep track of the last task and just yield task
        self.last_task = LLMTask(True, -1, -1)
        self.curr_task = None

        # forwarding
        while True:
            # working window reaches the end, convert to backwarding
            if curr_unit == self.num_units - 1:
                break

            for micro_step in range(self.micro_batch_per_block):
                curr_micro_batch = micro_step if self.top_down \
                    else self.micro_batch_per_block - 1 - micro_step
                curr_task = LLMTask(True, curr_micro_batch, curr_unit)
                self.curr_task = curr_task
                yield curr_task
                self.last_task = curr_task

            # reverse direction
            self.top_down = not self.top_down

            # move to next checkpoint
            curr_unit += 1

        # backwarding
        while True:
            # working window reaches the start, end the loop
            if curr_unit < 0:
                break

            for micro_step in range(self.micro_batch_per_block):
                curr_micro_batch = micro_step if self.top_down \
                    else self.micro_batch_per_block - 1 - micro_step
                curr_task = LLMTask(False, curr_micro_batch, curr_unit)
                self.curr_task = curr_task
                yield curr_task
                self.last_task = curr_task

            # reverse direction
            self.top_down = not self.top_down

            # move to next checkpoint
            curr_unit -= 1

    @property
    def new_unit_entered(self):
        curr_unit = self.curr_task.unit
        last_unit = self.last_task.unit
        return curr_unit != last_unit

    @property
    def first_unit_entered(self):
        return self.curr_task.unit == 0

    @property
    def last_unit_entered(self):
        return self.curr_task.unit == self.num_units - 1
