from .parameter import get_para_coordinator     # noqa: F401
from .interlayer import (                       # noqa: F401
    get_interlayer_coordinator,
    InterLayerTask,
    retrieve_tensor
)
from .optimizer import get_opt_coordinator      # noqa: F401

from flextrain.memory import get_data_stream
from flextrain.scheduler import (
    LLMTask,
    GreedySnakeBlockScheduler
)

# TEMP
from flextrain.utils import dist


def warmup_forward_pipeline(scheduler: GreedySnakeBlockScheduler):
    """
    Warm up the pipeline using both parameter and optimizer coordinator.
    """
    para = get_para_coordinator()
    opt = get_opt_coordinator()
    data_stream = get_data_stream()
    micro_batch_per_rank = para._micro_batch_per_rank

    # Reset the scheduler to the beginning.
    scheduler.reset()

    # Figure out the offset of the first task for the optimizer.
    opt_task_offset = opt.UPDATE_PARA + 1

    for i in range(opt_task_offset):
        third_next_task = scheduler.kth_next_forward_task(i)
        assert third_next_task is not None
        # We need to decrease the unit by 1 to align with the optimizer.
        third_next_task.unit -= 1
        dist.print_rank0(third_next_task)
        opt.pre_micro_batch_forward(None, third_next_task)
        data_stream.execute()
        data_stream.synchronize()
        dist.print_rank0()

    # Ensure the availability of the first unit.
    para._async_load_nvme_paras(0, 0)
    for i in range(micro_batch_per_rank):
        para._sync_nvme_operations(0, i)
        data_stream.synchronize()

        third_next_task = scheduler.kth_next_forward_task(i + opt_task_offset)
        opt.pre_micro_batch_forward(LLMTask(i, -1), third_next_task)
        para._async_load_nvme_paras(0, i + 1)
        para._async_load_gpu_paras(0, i)
        data_stream.execute()
        dist.print_rank0()

    if opt._finalized:
        exit(0)

    # Submit the next task.
    para._async_load_nvme_paras(1, 0)


def warmup_backward_pipeline(*args, **kwargs):
    """
    Warm up the backward pipeline.
    """
    para = get_para_coordinator()
    para.warmup_backward_pipeline()

    opt = get_opt_coordinator()
    opt.warmup_backward_pipeline()


def clear_backward_pipeline(*args, **kwargs):
    """
    Clear the backward pipeline.
    """
    para = get_para_coordinator()
    para.clear_backward_pipeline()

    opt = get_opt_coordinator()
    opt.clear_backward_pipeline()
