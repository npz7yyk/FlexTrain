from .parameter import get_para_coordinator     # noqa: F401
from .interlayer import (                       # noqa: F401
    get_interlayer_coordinator,
    InterLayerTask,
    retrieve_tensor
)
from .optimizer import get_opts_coordinator

from flextrain.memory import get_data_stream
from flextrain.scheduler import (
    LLMTask,
    GreedySnakeBlockScheduler
)


def warmup_forward_pipeline(scheduler: GreedySnakeBlockScheduler):
    """
    Warm up the pipeline of both parameter and optimizer coordinators.
    """
    para = get_para_coordinator()
    opts = get_opts_coordinator()
    data_stream = get_data_stream()
    micro_batch_per_rank = para._micro_batch_per_rank

    # Reset the scheduler to the beginning and enter the warmup stage.
    scheduler.enter_warmup_stage()
    tasks = iter(scheduler)

    # Figure out the offset of the first task for the optimizer.
    for i in range(opts.UPDATE_PARA + 1):
        next_task: LLMTask = next(tasks)

        # Synchronize inflight tasks.
        data_stream.synchronize()
        opts._sync_inflight_operations()

        opts._submit_micro_batch_task(
            True,
            next_task.unit,
            next_task.micro_batch
        )
        data_stream.execute()

    # Ensure the availability of the first unit.
    para._async_load_nvme_paras(0, micro_batch_per_rank - 1)
    for i in reversed(range(micro_batch_per_rank)):
        data_stream.synchronize()
        para._sync_nvme_operations(0, i)
        opts._sync_inflight_operations()

        # Ensure that two coordinators are aligned.
        opts._validate_forward_task(LLMTask(-1, i))

        para._async_load_nvme_paras(0, i - 1)
        para._async_load_gpu_paras(0, i)
        next_task: LLMTask = next(tasks)
        opts._submit_micro_batch_task(
            True,
            next_task.unit,
            next_task.micro_batch
        )
        data_stream.execute()

    # Submit the next task.
    para._async_load_nvme_paras(1, 0)

    # Reset the scheduler to the beginning and exit the warmup stage.
    scheduler.exit_warmup_stage()


def warmup_backward_pipeline(*args, **kwargs):
    """
    Warm up the backward pipeline.
    """
    para = get_para_coordinator()
    para.warmup_backward_pipeline()

    opts = get_opts_coordinator()
    opts.warmup_backward_pipeline()


def clear_backward_pipeline(*args, **kwargs):
    """
    Clear the backward pipeline.
    """
    para = get_para_coordinator()
    para.clear_backward_pipeline()

    opts = get_opts_coordinator()
    opts.clear_backward_pipeline()
