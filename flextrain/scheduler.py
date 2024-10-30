from dataclasses import dataclass


@dataclass
class LLMTask:
    """ FlexTrain LLM Task.

    Arguments:
        micro_batch (int): Micro batch index in the batch
        unit (int): Unit index in the LLM to be executed
    """
    micro_batch: int
    unit: int


class GreedySnakeBlockScheduler:
    """ FlexTrain Greedy Snake Block Scheduler.

    Arguments:
        num_micro_batch (int): Number of micro batches in the LLM
        num_unit (int): Number of units in the LLM
    """

    def __init__(
        self,
        num_micro_batch: int,
        num_unit: int
    ):
        self._in_forward = True
        self._num_micro_batch = num_micro_batch
        self._num_unit = num_unit
        self._num_fwd_task = num_micro_batch * (num_unit - 1)
        self._num_task = num_micro_batch * (2 * num_unit - 1)

    def __iter__(self):
        # Reset the task number
        if self._in_forward:
            self.cur_task_num = 0

        # Forward
        while True:
            yield self.curr_task
            self.cur_task_num += 1
            if self._in_forward:
                if self.cur_task_num == self._num_fwd_task:
                    self._in_forward = False
                    break
            else:
                if self.cur_task_num == self._num_task:
                    self._in_forward = True
                    break

    def _generate_task(self, cur_task_num):
        # Calculate the current unit
        cur_unit = cur_task_num // self._num_micro_batch
        is_forwarding = cur_unit < self._num_unit - 1
        if not is_forwarding:
            cur_unit = 2 * self._num_unit - 2 - cur_unit

        # Calculate the current micro batch
        cur_micro_batch = cur_task_num % self._num_micro_batch
        if cur_unit & 1:
            cur_micro_batch = self._num_micro_batch - 1 - cur_micro_batch
        else:
            cur_micro_batch = cur_micro_batch
        return LLMTask(cur_micro_batch, cur_unit)

    def reset(self):
        self.cur_task_num = 0

    @property
    def new_unit_entered(self):
        return self.cur_task_num % self._num_micro_batch == 0

    @property
    def in_first_unit(self):
        cur_unit = self.cur_task_num // self._num_micro_batch
        return cur_unit == 0 or cur_unit == 2 * self._num_unit - 2

    @property
    def in_last_unit(self):
        cur_unit = self.cur_task_num // self._num_micro_batch
        return cur_unit == self._num_unit - 1

    @property
    def curr_task(self):
        return self._generate_task(self.cur_task_num)

    @property
    def next_task(self):
        if self.cur_task_num == self._num_task - 1:
            return self._generate_task(0)
        else:
            return self._generate_task(self.cur_task_num + 1)

    def kth_next_forward_task(self, k):
        task_num = self.cur_task_num + k
        if task_num >= self._num_fwd_task:
            return None  # Out of bound, only used for forward pipeline
        else:
            return self._generate_task(self.cur_task_num + k)
