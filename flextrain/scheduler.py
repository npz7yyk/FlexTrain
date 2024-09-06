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
        self._micro_batch_per_block = micro_batch_per_block
        self._num_units = num_units
        self._ntasks = micro_batch_per_block * (2 * num_units - 1)

    def __iter__(self):
        self.cur_task_num = 0
        while True:
            yield self.curr_task
            self.cur_task_num += 1
            if self.cur_task_num == self._ntasks:
                break

    def _generate_task(self, cur_task_num):
        # Calculate the current unit
        cur_unit = cur_task_num // self._micro_batch_per_block
        is_forwarding = cur_unit < self._num_units - 1
        if not is_forwarding:
            cur_unit = 2 * self._num_units - 2 - cur_unit

        # Calculate the current micro batch
        cur_micro_batch = cur_task_num % self._micro_batch_per_block
        if cur_unit & 1:
            cur_micro_batch = self._micro_batch_per_block - 1 - cur_micro_batch
        else:
            cur_micro_batch = cur_micro_batch
        return LLMTask(is_forwarding, cur_micro_batch, cur_unit)

    @property
    def new_unit_entered(self):
        return self.cur_task_num % self._micro_batch_per_block == 0

    @property
    def in_last_micro_batch(self):
        return (self.cur_task_num + 1) % self._micro_batch_per_block == 0

    @property
    def in_first_unit(self):
        cur_unit = self.cur_task_num // self._micro_batch_per_block
        return cur_unit == 0 or cur_unit == 2 * self._num_units - 2

    @property
    def in_last_unit(self):
        cur_unit = self.cur_task_num // self._micro_batch_per_block
        return cur_unit == self._num_units - 1

    @property
    def last_task(self):
        if self.cur_task_num == 0:
            return self._generate_task(self._ntasks - 1)
        else:
            return self._generate_task(self.cur_task_num - 1)

    @property
    def curr_task(self):
        return self._generate_task(self.cur_task_num)

    @property
    def next_task(self):
        if self.cur_task_num == self._ntasks - 1:
            return self._generate_task(0)
        else:
            return self._generate_task(self.cur_task_num + 1)
