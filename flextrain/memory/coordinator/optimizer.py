import torch

from abc import ABC, abstractmethod
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from torch import Tensor
from tqdm import tqdm
from typing import Set, Dict

from flextrain.checkpointing import set_pre_backward_function
from flextrain.config import get_flextrain_config
from flextrain.memory import (
    Waitable,
    FunctionHandle,
    FlexTrainDataType as Dtype,
    FlexTrainDataID,
    RotateContainer,
    get_split_numels,
    allocate_memory_chunks,
    copy_segments,
    get_data_stream
)
from flextrain.memory.coordinator import (
    get_para_coordinator,
    get_interlayer_coordinator
)
from flextrain.memory.nvme_swapper import (
    NVMeGroup,
    _nvme_offload,
    _nvme_reload
)
from flextrain.scheduler import LLMTask
from flextrain.utils import dist
from flextrain.utils.logging import rank0_logger


@dataclass
class OptTarget(ABC):
    para: Tensor
    grad: Tensor


STEP_KEY = "step"


class FlexTrainCPUOptimizer(ABC):
    FWD_INDEX = 0
    BWD_INDEX = 1

    def __init__(self, unit_group_map: Dict[int, Dict]):
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._unit_group_map = unit_group_map

        # Initialize the state for each unit.
        if not hasattr(self, "state"):
            self.state = defaultdict(dict)
        for unit in unit_group_map:
            self.state[unit] = {STEP_KEY: 0}

    @abstractmethod
    def _step(self, step: int, args: Dict, opt_target: OptTarget):
        pass

    def update_state(self):
        for unit in self._unit_group_map:
            self.state[unit][STEP_KEY] += 1

    def _submit_micro_batch_step(
        self,
        unit_index: int,
        micro_batch_index: int,
        opt_target: OptTarget
    ):
        assert unit_index in self._unit_group_map, (
            "The unit index is not in the unit group map."
        )

        # Submit the step function to the executor.
        future = self._executor.submit(
            self._step,
            self.state[unit_index][STEP_KEY],
            self._unit_group_map[unit_index],
            opt_target
        )
        return FunctionHandle(future.result)

    @abstractmethod
    def submit_micro_batch_step(
        self, unit_index: int, micro_batch_index: int,
        para: torch.Tensor, grad: torch.Tensor,
        *args, **kwargs
    ) -> Waitable:
        pass


def _convert_dtype_view(
    tensor: Tensor,
    target_dtype: torch.dtype
) -> Tensor:
    if tensor.numel() == 0:
        assert tensor.shape[-1] == 0
        return torch.empty_like(tensor, dtype=target_dtype)
    return tensor.view(target_dtype)


@dataclass
class MicroBatchTask(LLMTask):
    forward: bool


class FlexTrainOptsCoordinator:
    TRANSFER_OPTS = 0
    RECOVER_GRAD = 0
    TRANSFER_GRAD = 0
    OPTIMIZER_STEP = 1
    OFFLOAD_GRAD = 1
    UPDATE_OPTS = 2
    UPDATE_PARA = 2
    OFFLOAD_PARA = 3

    def __init__(self):
        # Lazy initialization of optimizer coordinator.
        self._initialized = False

    @property
    def is_initialized(self):
        return self._initialized

    @property
    def _gpu_bwd_receive_grads(self):
        return self._gpu_bwd_grad_buffers[0]

    @property
    def _gpu_bwd_transfer_grads(self):
        return self._gpu_bwd_grad_buffers[1]

    @property
    def _cpu_opt_receive_grads(self):
        return self._cpu_opt_grad_buffers[0]

    @property
    def _cpu_opt_available_grads(self):
        return self._cpu_opt_grad_buffers[1]

    @property
    def _cpu_opt_receive_states(self):
        return self._cpu_opt_work_buffers[0]

    @property
    def _cpu_opt_available_states(self):
        return self._cpu_opt_work_buffers[1]

    @property
    def _cpu_opt_transfer_states(self):
        return self._cpu_opt_work_buffers[2]

    @property
    def _nvme_para_receive_buffer(self):
        return self._nvme_para_buffers[0]

    @property
    def _nvme_para_offload_buffer(self):
        return self._nvme_para_buffers[1]

    @property
    def _nvme_grad_receive_buffer(self):
        return self._nvme_grad_buffers[0]

    @property
    def _nvme_grad_offload_buffer(self):
        return self._nvme_grad_buffers[1]

    def _is_invalid_unit(self, unit_index: int):
        return unit_index < 0 or unit_index >= self._num_units

    def _is_invalid_micro_batch(self, micro_batch_index: int):
        return micro_batch_index < 0 or \
            micro_batch_index >= self._micro_batch_per_rank

    def _is_invalid_task(self, task: LLMTask):
        return task is None or \
            self._is_invalid_unit(task.unit) or \
            self._is_invalid_micro_batch(task.micro_batch)

    def initialize(
        self,
        cpu_optimizer: FlexTrainCPUOptimizer,
        opt_state_per_element: int
    ):
        # 0. Before initialization:
        # Ensure that the parameter coordinator is initialized.
        para = get_para_coordinator()
        assert para.is_initialized, (
            "Parameter coordinator must be initialized before init_optimizer."
        )
        self._para = para

        # Link the optimizer to the coordinator.
        self._optimizer = cpu_optimizer

        # Use the same data stream as the parameter & inter-layer coordinator.
        self._data_stream = get_data_stream()

        # 1. Set the configuration for the optimizer.
        self._num_units = para.num_units
        self._unit_parameters = para._unit_parameters
        # Master parameters are also regarded as a optimizer state.
        self._opt_state_per_element = opt_state_per_element + 1

        config = get_flextrain_config()

        assert config.split_ratio.optimizer[0] == 0., (
            "FlexTrain optimizer currently does not support GPU optimizer. "
            "Please set the GPU optimizer ratio to 0. "
        )
        # Drop the GPU optimizer ratio.
        opts_cpu_nvme_ratio = config.split_ratio.optimizer[1:]

        # Configuration for mixed precision.
        device_dtype = config.mixed_precision.device_dtype
        gradacc_dtype = config.mixed_precision.gradacc_dtype
        master_dtype = config.mixed_precision.master_dtype
        self._gradacc_dtype_incompatible = device_dtype != gradacc_dtype

        # Configuration for optimizer partition.
        self._alpha = config.split_ratio.alpha[0]
        self._unit_numel = para._aligned_unit_numel
        self._micro_batch_per_rank = para._micro_batch_per_rank

        self._mb_gpu_para_alpha_splits = para._micro_batch_gpu_alpha_splits
        self._mb_cpu_para_alpha_splits = para._micro_batch_cpu_alpha_splits
        self._mb_nvme_para_alpha_splits = para._micro_batch_nvme_alpha_splits

        # Calculate the parameter numels of forward and backward.
        self._forward_numel = \
            self._mb_gpu_para_alpha_splits[0] + \
            self._mb_cpu_para_alpha_splits[0] + \
            self._mb_nvme_para_alpha_splits[0]
        self._backward_numel = \
            self._mb_gpu_para_alpha_splits[1] + \
            self._mb_cpu_para_alpha_splits[1] + \
            self._mb_nvme_para_alpha_splits[1]
        max_numel = max(self._forward_numel, self._backward_numel)
        # Calculate the numel of forward and backward optimizer states.
        self._forward_opt_numel = self._forward_numel * \
            self._opt_state_per_element
        self._backward_opt_numel = self._backward_numel * \
            self._opt_state_per_element

        # How to split the micro-batch parameters.
        self._para_splits = [
            self._mb_cpu_para_alpha_splits[0],
            self._mb_cpu_para_alpha_splits[1],
            self._mb_gpu_para_alpha_splits[0],
            self._mb_gpu_para_alpha_splits[1],
            self._mb_nvme_para_alpha_splits[0],
            self._mb_nvme_para_alpha_splits[1]
        ]
        # How to split the forward parameters in optimizer.
        self._forward_para_splits = [
            self._mb_cpu_para_alpha_splits[0],
            self._mb_gpu_para_alpha_splits[0],
            self._mb_nvme_para_alpha_splits[0]
        ]
        # How to split the backward parameters in optimizer.
        self._backward_para_splits = [
            self._mb_cpu_para_alpha_splits[1],
            self._mb_gpu_para_alpha_splits[1],
            self._mb_nvme_para_alpha_splits[1]
        ]

        # How to split the forward optimizer states across devices.
        self._forward_opt_splits = get_split_numels(
            self._forward_opt_numel,
            opts_cpu_nvme_ratio, num_levels=2
        )
        # How to split the backward optimizer states across devices.
        self._backward_opt_splits = get_split_numels(
            self._backward_opt_numel,
            opts_cpu_nvme_ratio, num_levels=2
        )

        # NVMe group for optimizer.
        self._opt_nvme_group = NVMeGroup([
            self._forward_opt_splits[1],
            self._backward_opt_splits[1]
        ])
        # End of configuration.

        # 2. Allocate memory for the optimizer.
        # Used for accumulating / transferring backward gradients.
        self._gpu_bwd_grad_buffers = RotateContainer(
            allocate_memory_chunks(
                self._unit_numel, 2, gradacc_dtype,
                torch.cuda.current_device()
            )
        )

        # If gradacc_dtype is different from device_dtype,
        # we need an extra buffer for backward gradients.
        if self._gradacc_dtype_incompatible:
            self._gpu_bwd_extra_grads = torch.empty(
                self._unit_numel, dtype=device_dtype,
                device=torch.cuda.current_device()
            )

        # Used for receiving gradients / working with optimizer.
        self._cpu_opt_grad_buffers = RotateContainer(
            allocate_memory_chunks(
                max_numel, 2, master_dtype, torch.device('cpu')
            )
        )

        # Optimizer working buffer.
        self._cpu_opt_work_buffers = RotateContainer(
            allocate_memory_chunks(
                max_numel * self._opt_state_per_element,
                3, master_dtype, torch.device('cpu')
            )
        )

        # NVMe parameter offload buffer.
        self._nvme_para_buffers = RotateContainer(
            allocate_memory_chunks(
                max_numel, 2, device_dtype, torch.device('cpu')
            )
        )
        self._para_nvme_group = para._nvme_group

        # CPU optimizer base.
        self._cpu_opt_fwd_base = allocate_memory_chunks(
            self._forward_opt_splits[0], (
                self._num_units,
                self._micro_batch_per_rank
            ),
            master_dtype, torch.device('cpu')
        ).zero_()
        self._cpu_opt_bwd_base = allocate_memory_chunks(
            self._backward_opt_splits[0], (
                self._num_units,
                self._micro_batch_per_rank
            ),
            master_dtype, torch.device('cpu')
        ).zero_()
        # End of memory allocation.

        # 3. Initialize master parameters from device parameters.
        gpu_base = para._gpu_para_base
        cpu_base = para._cpu_para_base
        nvme_src = para._nvme_available_paras
        temp_cpu_buffer: Tensor = self._cpu_opt_available_states

        dist.barrier()
        units = tqdm(
            range(self._num_units), desc="FlexTrain Opt. Init."
        ) if dist.get_rank() == 0 else range(self._num_units)
        for unit in units:
            for micro_batch in range(self._micro_batch_per_rank):
                gpu_src = gpu_base[unit][micro_batch]
                gpu_fwd_src, gpu_bwd_src = torch.split(
                    gpu_src, self._mb_gpu_para_alpha_splits
                )
                cpu_src = cpu_base[unit][micro_batch]
                cpu_fwd_src, cpu_bwd_src = torch.split(
                    cpu_src, self._mb_cpu_para_alpha_splits
                )
                self._para_nvme_group.group_reload(
                    FlexTrainDataID(unit, micro_batch, Dtype.PARA),
                    nvme_src, async_op=False
                )
                nvme_fwd_src, nvme_bwd_src = torch.split(
                    nvme_src, self._mb_nvme_para_alpha_splits
                )

                # 1. Copy the forward parameters.
                temp_cpu_buffer.zero_()

                # Locate the target memory.
                tar_fwd_mem = temp_cpu_buffer[:self._forward_numel]
                cpu_fwd_tar, gpu_fwd_tar, nvme_fwd_tar = torch.split(
                    tar_fwd_mem, self._forward_para_splits
                )

                # Copy parameters from three sources.
                gpu_fwd_tar.copy_(gpu_fwd_src)
                cpu_fwd_tar.copy_(cpu_fwd_src)
                nvme_fwd_tar.copy_(nvme_fwd_src)

                # Locate the source memory.
                fwd_src = temp_cpu_buffer[:self._forward_opt_numel]
                cpu_fwd_src, nvme_fwd_src = torch.split(
                    fwd_src, self._forward_opt_splits
                )

                # Locate the target memory.
                cpu_fwd_tar = self._cpu_opt_fwd_base[unit][micro_batch]

                # Copy the parameters.
                cpu_fwd_tar.copy_(cpu_fwd_src)
                self._opt_nvme_group.single_offload(
                    FlexTrainDataID(unit, micro_batch, Dtype.OPTS),
                    nvme_fwd_src, index=0, async_op=False
                )
                # End of forward parameters.

                # 2. Copy the backward parameters.
                temp_cpu_buffer.zero_()

                # Locate the target memory.
                tar_bwd_mem = temp_cpu_buffer[:self._backward_numel]
                cpu_bwd_tar, gpu_bwd_tar, nvme_bwd_tar = torch.split(
                    tar_bwd_mem, self._backward_para_splits
                )

                # Copy parameters from three sources.
                gpu_bwd_tar.copy_(gpu_bwd_src)
                cpu_bwd_tar.copy_(cpu_bwd_src)
                nvme_bwd_tar.copy_(nvme_bwd_src)

                # Locate the source memory.
                bwd_src = temp_cpu_buffer[:self._backward_opt_numel]
                cpu_bwd_src, nvme_bwd_src = torch.split(
                    bwd_src, self._backward_opt_splits
                )

                # Locate the target memory.
                cpu_bwd_tar = self._cpu_opt_bwd_base[unit][micro_batch]

                # Copy the parameters.
                cpu_bwd_tar.copy_(cpu_bwd_src)
                self._opt_nvme_group.single_offload(
                    FlexTrainDataID(unit, micro_batch, Dtype.OPTS),
                    nvme_bwd_src, index=1, async_op=False
                )
                # End of backward parameters.
        # End of initialization.

        # Inflight task queue, each element is either MircroBatchTask or None.
        # Tasks in the queue are (in order, fwd / bwd):
        # 1. transfer_opts + recover_grad / transfer_grad
        # 2. optimizer_step + None / offload_grad
        # 3. update_opts + update_para
        # 4. None / para_offload
        self._inflight_task_queue: deque[MicroBatchTask] = deque(maxlen=4)
        self._inflight_task_set: Set[Waitable] = set()

    def _initialize_alpha_split(self):
        # If the coordinator is already initialized, return.
        if self._initialized:
            return
        self._initialized = True

        # Ensure that the parameter coordinator is initialized.
        para = get_para_coordinator()
        assert para.is_initialized, (
            "Parameter coordinator must be initialized before init_optimizer."
        )
        # Ensure that the interlayer coordinator is initialized.
        interlayer = get_interlayer_coordinator()
        assert interlayer.is_initialized, (
            "Interlayer coordinator must be initialized before init_optimizer."
        )

        # Figure out the numel each buffer can hold.
        cpu_grad_buffer1_numel = interlayer._ckpt_numels[1]

        bwd_cpu_opt_cpu_para_overlap = min(
            self._backward_opt_splits[0],
            self._mb_cpu_para_alpha_splits[1]
        )
        # Fwd CPU buffer + bwd CPU overlap.
        cpu_grad_buffer2_numel = \
            self._mb_cpu_para_alpha_splits[0] + \
            bwd_cpu_opt_cpu_para_overlap

        gpu_grad_buffer1_numel = interlayer._ckpt_numels[0]

        bwd_cpu_opt_gpu_para_overlap = min(
            self._backward_opt_splits[0] - bwd_cpu_opt_cpu_para_overlap,
            self._mb_gpu_para_alpha_splits[1]
        )
        # Fwd GPU buffer + bwd GPU overlap.
        gpu_grad_buffer2_numel = \
            self._mb_gpu_para_alpha_splits[0] + \
            bwd_cpu_opt_gpu_para_overlap

        # Total forward gradient numel.
        forward_grad_numel = self._forward_numel

        # Consider dtype incompatibility.
        config = get_flextrain_config()
        device_itemsize = config.mixed_precision.device_dtype.itemsize
        gradacc_itemsize = config.mixed_precision.gradacc_dtype.itemsize
        ratio = gradacc_itemsize // device_itemsize

        # Assign memory buffers and update plans.
        # Followings are the most complicated part of this repository.

        # cvtd = converted
        cvtd_cpu_grad_buffer1_numel = cpu_grad_buffer1_numel // ratio
        cvtd_cpu_grad_buffer2_numel = cpu_grad_buffer2_numel // ratio
        cvtd_gpu_grad_buffer1_numel = gpu_grad_buffer1_numel // ratio
        cvtd_gpu_grad_buffer2_numel = gpu_grad_buffer2_numel // ratio
        numels = [
            cvtd_cpu_grad_buffer1_numel, cvtd_cpu_grad_buffer2_numel,
            cvtd_gpu_grad_buffer1_numel, cvtd_gpu_grad_buffer2_numel
        ]

        cpu_buffer1_needed_numel = cvtd_cpu_grad_buffer1_numel
        cpu_buffer2_needed_numel = cvtd_cpu_grad_buffer2_numel
        gpu_buffer1_needed_numel = cvtd_gpu_grad_buffer1_numel
        gpu_buffer2_needed_numel = cvtd_gpu_grad_buffer2_numel
        nvme_numel = forward_grad_numel - sum(numels[:4])
        if forward_grad_numel < sum(numels[:1]):
            cpu_buffer1_needed_numel = forward_grad_numel - sum(numels[:0])
            cpu_buffer2_needed_numel = 0
            gpu_buffer1_needed_numel = 0
            gpu_buffer2_needed_numel = 0
            nvme_numel = 0
        elif forward_grad_numel < sum(numels[:2]):
            cpu_buffer2_needed_numel = forward_grad_numel - sum(numels[:1])
            gpu_buffer1_needed_numel = 0
            gpu_buffer2_needed_numel = 0
            nvme_numel = 0
        elif forward_grad_numel < sum(numels[:3]):
            gpu_buffer1_needed_numel = forward_grad_numel - sum(numels[:2])
            gpu_buffer2_needed_numel = 0
            nvme_numel = 0
        elif forward_grad_numel < sum(numels[:4]):
            gpu_buffer2_needed_numel = forward_grad_numel - sum(numels[:3])
            nvme_numel = 0
        else:
            pass

        def _create_view(tensor: Tensor, numel: int, dtype: torch.dtype):
            if isinstance(tensor, list):
                assert all(isinstance(t, Tensor) for t in tensor)
                return [_create_view(t, numel, dtype) for t in tensor]
            return _convert_dtype_view(tensor, dtype)[..., :numel]

        self._cpu_grad_buffer1 = _create_view(
            interlayer.cpu_ckpt_base, cpu_buffer1_needed_numel,
            config.mixed_precision.gradacc_dtype
        )
        self._cpu_grad_buffer2 = _create_view(
            para._cpu_para_base, cpu_buffer2_needed_numel,
            config.mixed_precision.gradacc_dtype
        )
        self._gpu_grad_buffer1 = _create_view(
            interlayer.gpu_ckpt_base, gpu_buffer1_needed_numel,
            config.mixed_precision.gradacc_dtype
        )
        self._gpu_grad_buffer2 = _create_view(
            para._gpu_para_base, gpu_buffer2_needed_numel,
            config.mixed_precision.gradacc_dtype
        )

        # How to reconstruct the forward gradients.
        self._forward_mem_splits = [
            cpu_buffer1_needed_numel, cpu_buffer2_needed_numel,
            gpu_buffer1_needed_numel, gpu_buffer2_needed_numel,
            nvme_numel
        ]

        # How to update the backward cpu parameters.
        cpu_para_borrowed_numel = cpu_buffer2_needed_numel * ratio
        self._backward_update_cpu_splits = [
            cpu_para_borrowed_numel,
            self._mb_cpu_para_alpha_splits[1] - cpu_para_borrowed_numel
        ]
        # How to update the backward gpu parameters.
        gpu_para_borrowed_numel = gpu_buffer2_needed_numel * ratio
        self._backward_update_gpu_splits = [
            gpu_para_borrowed_numel,
            self._mb_gpu_para_alpha_splits[1] - gpu_para_borrowed_numel
        ]
        # How to split the backward parameters in optimizer.
        self._cpu_opt_para_update_splits = [
            *self._backward_update_cpu_splits,
            *self._backward_update_gpu_splits,
            self._mb_nvme_para_alpha_splits[1]
        ]

        # NVMe gradient buffers.
        self._nvme_grad_buffers = RotateContainer(
            allocate_memory_chunks(
                nvme_numel, 2, config.mixed_precision.gradacc_dtype,
                torch.device('cpu')
            )
        )
        # End of assignment. Incredible!!!

        # Log the configuration.
        rank0_logger.info(
            "\n\n> "
            f"FlexTrain optimizer coordinator initialized "
            f"with configurations:\n"
            f"  - Parameter alpha split ratio (forward, backward): "
            f"({self._alpha}, {1 - self._alpha})\n"
            f"  - Parameter alpha split numels (forward, backward): "
            f"({self._forward_numel}, {self._backward_numel})\n"
            f"  - Forward parameter split numels (CPU, GPU, NVMe): "
            f"{self._forward_para_splits}\n"
            f"  - Forward optimizer split numels (CPU, NVMe): "
            f"{self._forward_opt_splits}\n"
            f"  - Backward parameter split numels (CPU, GPU, NVMe): "
            f"{self._backward_para_splits}\n"
            f"  - Backward optimizer split numels (CPU, NVMe): "
            f"{self._backward_opt_splits}\n"
            f"  - Forward gradient numel: {forward_grad_numel}\n"
            f"  - Gradient dtype itemsize / device dtype itemsize: {ratio}\n"
            f"  - Checkpoint borrowable numels (CPU1, GPU1): "
            f"({cvtd_cpu_grad_buffer1_numel}, {cvtd_gpu_grad_buffer1_numel})\n"
            f"  - Parameter borrowable numels (CPU2, GPU2): "
            f"({cvtd_cpu_grad_buffer2_numel}, {cvtd_gpu_grad_buffer2_numel})\n"
            f"  - Gradient buffer numels (CPU1, CPU2, GPU1, GPU2, NVMe): "
            f"{self._forward_mem_splits}\n"
        )

        # TEMP
        self._data_stream = get_data_stream()
        self._grad_partition = allocate_memory_chunks(
            self._forward_numel + self._backward_numel, (self._num_units, self._micro_batch_per_rank),
            config.mixed_precision.gradacc_dtype, torch.cuda.current_device()
        )

    def _calculate_global_grad_norm(self):
        # Calculate the global gradient norm.
        global_grad_norm = torch.tensor([0.], device=torch.cuda.current_device())
        for unit in reversed(range(self._num_units)):
            global_grad_norm += self._grad_partition[unit].norm() ** 2
            dist.print_rank0(f"Rank {dist.get_rank()} layer {unit} grad norm: {global_grad_norm.item()}")
        # torch.save(self._grad_partition, f"/shared_ssd_storage/yikang/FlexTrain/logs/grad2/grad_norm_{dist.get_rank()}.pt")
        dist.all_reduce(global_grad_norm, op=dist.ReduceOp.SUM)
        dist.print_rank0(global_grad_norm.item())
        return global_grad_norm.item()

    def _submit_micro_batch_task(
        self,
        forward: bool,
        unit_index: int,
        micro_batch_index: int
    ):
        # If optimizer coordinator is not ready, return.
        if not self.is_initialized:
            return

        # Build micro-batch task.
        task = MicroBatchTask(unit_index, micro_batch_index, forward)

        # If the task is invalid, add placeholder to the queue.
        if self._is_invalid_task(task):
            self._inflight_task_queue.appendleft(None)
        else:
            # Submit the task to the task queue.
            self._inflight_task_queue.appendleft(task)

        # Submit general tasks.
        self._submit_transfer_opts()
        self._submit_optimizer_step()
        self._submit_update_opts()
        self._submit_update_para()

        if task.forward:
            # Submit gradient recover task.
            self._submit_recover_grad()
        else:
            # Submit gradient offload task.
            self._submit_offload_grad()
            # Submit offload parameter task.
            self._submit_offload_para()

            # Submit gradient transfer task.
            def pre_micro_batch_backward_task():
                self._submit_transfer_grad()
                self._data_stream.execute()

            # Trick: execute gradient reduce-scatter after recomputation.
            # For the first unit, submit the task directly.
            if task.unit:
                set_pre_backward_function(pre_micro_batch_backward_task)
            else:
                self._submit_transfer_grad()

    def _sync_inflight_operations(self, skip_rotate_buffers=False):
        # If optimizer coordinator is not ready, return.
        if not self.is_initialized:
            return

        # Synchronize the inflight tasks.
        for task in self._inflight_task_set:
            # Synchronize the task.
            task.wait()

        # Clear the inflight task set.
        self._inflight_task_set.clear()

        if skip_rotate_buffers:
            return

        dist.print_rank0("\nRotate buffers.")

        # Rotate micro-batch buffers.
        self._cpu_opt_grad_buffers.rotate()
        self._cpu_opt_work_buffers.rotate()
        self._nvme_para_buffers.rotate()
        self._nvme_grad_buffers.rotate()

    def _prepare_unit_grads(self, unit_index: int):
        """
        Prepare the gradients for the unit.
        The gradients are prepared in _gpu_available_grads.

        Args:
            unit_index (int): Index of the unit.

        Returns:
            None
        """
        # Check if the unit is valid.
        if self._is_invalid_unit(unit_index):
            return

        # Get the unit parameters.
        unit_paras = self._unit_parameters[unit_index]

        # Get the gradient buffer.
        if self._gradacc_dtype_incompatible:
            grad_buffer = self._gpu_bwd_extra_grads
        else:
            grad_buffer = self._gpu_bwd_receive_grads

        # Zero the receive buffer.
        torch.zero_(self._gpu_bwd_receive_grads)

        # Link the gradients.
        unit_paras.link_grad_to(grad_buffer)

    def _submit_transfer_grad(self):
        """ Launch the async IO operation to transfer gradients. """
        # Locate and unpack the task.
        task = self._inflight_task_queue[self.TRANSFER_GRAD]
        # Return if no task to conduct.
        if task is None:
            return
        dist.print_rank0("Transfer grads:", task)
        unit_index, micro_batch_index = task.unit, task.micro_batch

        # 1. Locate the target memory.
        # 2. Conduct all-reduce into tensor if necessary.
        default_stream = torch.cuda.current_stream()

        def transfer_grads():
            # Locate the target memory.
            src_full_grads = torch.chunk(
                self._gpu_bwd_transfer_grads, self._micro_batch_per_rank
            )[micro_batch_index]
            mem_partition = torch.chunk(
                src_full_grads, dist.get_world_size()
            )[dist.get_rank()]

            # Synchronize with the default stream for the first unit.
            # Because there is no torch.cuda.synchronize() before.
            if unit_index == 0:
                default_stream.synchronize()

            # All-reduce the gradients.
            dist.reduce_scatter(
                mem_partition, src_full_grads,
                dist.ReduceOp.AVG
            )

            # Split the mem_partition.
            fwd_cpu, bwd_cpu, fwd_gpu, bwd_gpu, fwd_nvme, bwd_nvme = \
                torch.split(mem_partition, self._para_splits)

            # Store forward gradients into buffers.
            copy_segments(
                [fwd_cpu, fwd_gpu, fwd_nvme],
                [
                    self._cpu_grad_buffer1[unit_index][micro_batch_index],
                    self._cpu_grad_buffer2[unit_index][micro_batch_index],
                    self._gpu_grad_buffer1[unit_index][micro_batch_index],
                    self._gpu_grad_buffer2[unit_index][micro_batch_index],
                    self._nvme_grad_receive_buffer
                ]
            )
            # Move backward gradients into working buffer.
            bwd_cpu_tar, bwd_gpu_tar, bwd_nvme_tar = torch.split(
                self._cpu_opt_receive_grads[:self._backward_numel],
                self._backward_para_splits
            )
            bwd_cpu_tar.copy_(bwd_cpu, non_blocking=True)
            bwd_gpu_tar.copy_(bwd_gpu, non_blocking=True)
            bwd_nvme_tar.copy_(bwd_nvme, non_blocking=True)

            self._grad_partition[unit_index][micro_batch_index].copy_(mem_partition)

        # Submit the task to the data stream.
        self._data_stream.submit(transfer_grads)

    def _submit_recover_grad(self):
        """ Launch the async IO operation to recover forward gradients. """
        # Locate and unpack the task.
        task = self._inflight_task_queue[self.RECOVER_GRAD]
        # Return if no task to conduct.
        if task is None:
            return
        dist.print_rank0("Recover grads:", task)
        return

        # Copy the forward gradients from borrowed memory into working buffer.
        # 1. Locate the source memory.
        cpu_src1 = self._cpu_grad_buffer1[unit_index][micro_batch_index]
        cpu_src2 = self._cpu_grad_buffer2[unit_index][micro_batch_index]
        gpu_src1 = self._gpu_grad_buffer1[unit_index][micro_batch_index]
        gpu_src2 = self._gpu_grad_buffer2[unit_index][micro_batch_index]

        # 2. Locate the target memory.
        cpu_tar1, cpu_tar2, gpu_tar1, gpu_tar2, nvme_tar = torch.split(
            self._cpu_opt_receive_grads[:self._forward_numel],
            self._forward_mem_splits
        )

        # 3. Copy the source memory to the target memory.
        def recover_grads():
            cpu_tar1.copy_(cpu_src1, non_blocking=True)
            cpu_tar2.copy_(cpu_src2, non_blocking=True)
            gpu_tar1.copy_(gpu_src1, non_blocking=True)
            gpu_tar2.copy_(gpu_src2, non_blocking=True)

        handle = _nvme_reload(
            FlexTrainDataID(unit_index, Dtype.GRAD),
            nvme_tar, async_op=True
        )

        # 4. Submit the task to the data stream.
        self._data_stream.submit(recover_grads)

    def _submit_transfer_opts(self):
        """ Launch the async IO operation to transfer optimizer states. """
        # Locate and unpack the task.
        task = self._inflight_task_queue[self.TRANSFER_OPTS]
        # Return if no task to conduct.
        if task is None:
            return
        dist.print_rank0("Transfer opts:", task)
        return

        # dist.print_rank0(f"Submitting transfer opts unit {unit_index} micro_batch {micro_batch_index}")
        # 1. Locate the source memory.
        cpu_src = self._cpu_opt_fwd_base if forward else self._cpu_opt_bwd_base
        cpu_src = cpu_src[unit_index][micro_batch_index]

        # 2. Locate the target memory.
        receive_buffer = self._cpu_opt_receive_states
        clip_numel = self._forward_opt_numel \
            if forward else self._backward_opt_numel
        receive_buffer = receive_buffer[:clip_numel]
        cpu_tar, nvme_tar = torch.split(
            receive_buffer,
            self._forward_opt_splits if forward else self._backward_opt_splits
        )

        # 3. Copy the source memory to the target memory.
        cpu_tar.copy_(cpu_src, non_blocking=True)
        return self._opt_nvme_group.single_reload(
            FlexTrainDataID(unit_index, micro_batch_index, Dtype.OPTS),
            nvme_tar, index=0 if forward else 1, async_op=True
        )

    def _submit_optimizer_step(self):
        """ Launch the async optimizer step operation. """
        # Locate and unpack the task.
        task = self._inflight_task_queue[self.OPTIMIZER_STEP]
        # Return if no task to conduct.
        if task is None:
            return
        dist.print_rank0("Optimizer step:", task)
        return

        # 1. Locate the parameters / gradients / optimizer states.
        clip_numel = self._forward_opt_numel \
            if forward else self._backward_opt_numel
        cpu_states = torch.chunk(
            self._cpu_opt_available_states[:clip_numel],
            self._opt_state_per_element
        )
        cpu_paras = cpu_states[0]
        cpu_grads = self._cpu_opt_available_grads
        cpu_states = cpu_states[1:]

        # 2. Submit the optimizer step task.
        handle = self._optimizer.submit_micro_batch_step(
            unit_index, micro_batch_index,
            cpu_paras, cpu_grads, *cpu_states
        )

    def _submit_offload_grad(self):
        """ Launch the async IO operation to offload forward gradients. """
        # Locate and unpack the task.
        task = self._inflight_task_queue[self.OFFLOAD_GRAD]
        # Return if no task to conduct.
        if task is None:
            return
        dist.print_rank0("Offload grads:", task)
        return

        # 1. Submit the optimizer step task.
        handle = self._grad_nvme_group.single_offload(
            FlexTrainDataID(unit_index, Dtype.GRAD),
            self._nvme_grad_offload_buffer,
            index=micro_batch_index, async_op=True
        )

    def _submit_update_opts(self):
        """ Launch the async IO operation to update optimizer states. """
        # Locate and unpack the task.
        task = self._inflight_task_queue[self.UPDATE_OPTS]
        # Return if no task to conduct.
        if task is None:
            return
        dist.print_rank0("Update opts:", task)
        return

        # First, write back the optimizer states.
        # 1. Locate the source memory.
        writeback_buffer = self._cpu_opt_transfer_states
        clip_numel = self._forward_opt_numel \
            if forward else self._backward_opt_numel
        writeback_buffer = writeback_buffer[:clip_numel]
        cpu_src, nvme_src = torch.split(
            writeback_buffer,
            self._forward_opt_splits if forward else self._backward_opt_splits
        )

        # 2. Locate the target memory.
        cpu_tar = self._cpu_opt_fwd_base if forward else self._cpu_opt_bwd_base
        cpu_tar = cpu_tar[unit_index][micro_batch_index]

        # 3. Copy the source memory to the target memory.
        cpu_tar.copy_(cpu_src, non_blocking=True)
        handle1 = self._opt_nvme_group.single_offload(
            FlexTrainDataID(unit_index, Dtype.OPTS),
            nvme_src, index=0 if forward else 1, async_op=True
        )

        # Second, update the parameters.
        # 1. Locate the source memory.
        updated_para = torch.chunk(
            writeback_buffer, self._opt_state_per_element
        )[0]

        # 2. Locate the target memory.
        cpu_para = self._para._cpu_para_base[unit_index][micro_batch_index]
        cpu_fwd_tar, cpu_bwd_tar = torch.split(
            cpu_para, self._mb_cpu_para_alpha_splits
        )
        gpu_para = self._para._gpu_para_base[unit_index][micro_batch_index]
        gpu_fwd_tar, gpu_bwd_tar = torch.split(
            gpu_para, self._mb_gpu_para_alpha_splits
        )

        if forward:
            # 1. Locate the source memory.
            cpu_src, gpu_src, nvme_src = torch.split(
                updated_para, self._forward_para_splits
            )
            cpu_fwd_tar.copy()
        else:
            # 1. Locate the source memory.
            _, cpu_src, _, gpu_src, _ = torch.split(
                updated_para, self._cpu_opt_para_update_splits
            )

    def _submit_update_para(self):
        """ Launch the async IO operation to update parameters. """
        # Locate and unpack the task.
        task = self._inflight_task_queue[self.UPDATE_PARA]
        # Return if no task to conduct.
        if task is None:
            return
        dist.print_rank0("Update para:", task)
        return

        # 1. Locate the source memory.
        updated_para = self._cpu_opt_transfer_states
        clip_numel = self._forward_numel if forward else self._backward_numel
        updated_para = updated_para[:clip_numel]
        cpu_src, gpu_src, nvme_src = torch.split(
            updated_para, self._forward_para_splits
        )

        # 2. Locate the target memory.
        cpu_para = self._para._cpu_para_base[unit_index][micro_batch_index]
        cpu_fwd_tar, cpu_bwd_tar = torch.split(
            cpu_para, self._mb_cpu_para_alpha_splits
        )
        gpu_para = self._para._gpu_para_base[unit_index][micro_batch_index]
        gpu_fwd_tar, gpu_bwd_tar = torch.split(
            gpu_para, self._mb_gpu_para_alpha_splits
        )
        nvme_para = self._para._nvme_available_paras

        # 3. Copy the source memory to the target memory.
        cpu_fwd_tar.copy_(cpu_src, non_blocking=True)
        gpu_fwd_tar.copy_(gpu_src, non_blocking=True)
        nvme_para.copy_(nvme_src, non_blocking=True)

    def _submit_offload_para(self):
        """ Launch the async IO operation to offload parameters. """
        # Locate and unpack the task.
        task = self._inflight_task_queue[self.OFFLOAD_PARA]
        # Return if no task to conduct.
        if task is None:
            return
        dist.print_rank0("Para offload:", task)
        return

        # 1. Locate the source memory.
        src_mem = self._nvme_para_offload_buffer
        clip_numel = self._forward_numel if forward else self._backward_numel
        src_mem = src_mem[:clip_numel]

        # 2. Submit the offload task.
        handle = self._para_nvme_group.single_offload(
            FlexTrainDataID(unit_index, micro_batch_index, Dtype.PARA),
            src_mem, index=0 if forward else 1, async_op=True
        )

    def _validate_forward_task(self, curr_task: LLMTask):
        # For the first iteration, the optimizer is not ready.
        if not self.is_initialized:
            return

        # Find the just finished parameter update task.
        last_para_update = self._inflight_task_queue[self.UPDATE_PARA]
        # Unpack the current task.
        curr_unit = curr_task.unit
        curr_micro_batch = curr_task.micro_batch
        assert last_para_update is not None
        assert last_para_update.forward
        assert last_para_update.unit == curr_unit + 1, (
            f"Last para update unit {last_para_update.unit} "
            f"!= current unit {curr_unit + 1}"
        )
        assert last_para_update.micro_batch == curr_micro_batch, (
            f"Last para update micro-batch {last_para_update.micro_batch} "
            f"!= current micro-batch {curr_micro_batch}"
        )

    def pre_micro_batch_forward(
        self, curr_task: LLMTask, third_next_task: LLMTask
    ):
        # For the first iteration, the optimizer is not ready.
        if not self.is_initialized:
            return

        # Synchronize inflight tasks.
        self._sync_inflight_operations()

        # Check if the last optimizer step is valid.
        self._validate_forward_task(curr_task)

        # Submit a new task to the inflight task queue.
        self._submit_micro_batch_task(
            True,
            third_next_task.unit + 1,
            third_next_task.micro_batch
        )

    def pre_micro_batch_backward(self, curr_task: LLMTask):
        """ Submit tasks for the given micro-batch in backward pass.

        Args:
            unit_index (int): Index of the unit.
            micro_batch_index (int): Index of the micro-batch.

        Returns:
            None
        """
        # Zero the extra buffer if necessary.
        if self._gradacc_dtype_incompatible:
            torch.zero_(self._gpu_bwd_extra_grads)

        # Synchronize inflight tasks.
        self._sync_inflight_operations()

        # Submit a new task to the inflight task queue.
        self._submit_micro_batch_task(
            False,
            curr_task.unit + 1,
            curr_task.micro_batch
        )

    def post_micro_batch_backward(self, curr_task: LLMTask):
        """ Conduct post-processing after the backward of the micro-batch.

        Args:
            unit_index (int): Index of the unit.
            micro_batch_index (int): Index of the micro-batch.

        Returns:
            None
        """
        # Check if the unit is valid.
        if self._is_invalid_task(curr_task):
            assert False
            return

        # If the gradients are not compatible with the device dtype,
        # we need explicitly accumulate the gradients.
        if self._gradacc_dtype_incompatible:
            gradacc_buffer = self._gpu_bwd_receive_grads
            gradacc_buffer += self._gpu_bwd_extra_grads

    def pre_unit_forward(self, unit_index: int):
        pass

    def pre_unit_backward(self, unit_index: int):
        """ Prepare the unit for backward pass.

        Functions:
        1. Ensure the availability of the gradients buffer.

        Args:
            unit_index (int): Index of the unit.

        Returns:
            None
        """
        # Check if the unit is valid.
        if self._is_invalid_unit(unit_index):
            return

        # Prepare the gradient buffer.
        self._gpu_bwd_grad_buffers.rotate()
        self._prepare_unit_grads(unit_index)

    def warmup_backward_pipeline(self):
        # Complete the initialization.
        self._initialize_alpha_split()
        # Inform the parameter coordinator that future updates are coming.
        self._para.parameter_updated = True

        # Flush the inflight task queue.
        for _ in range(4):
            self._inflight_task_queue.appendleft(None)
        # Clear the inflight task set.
        self._inflight_task_set.clear()

    def clear_backward_pipeline(self):
        """
        Cleanup the backward pipeline.
        """
        # Move the gradient of the first unit to transfer_buffer.
        self._gpu_bwd_grad_buffers.rotate()

        # Conduct the optimizer step of the first unit.
        for mb in reversed(range(self._micro_batch_per_rank)):
            self._data_stream.synchronize()
            self._sync_inflight_operations()

            # Unpack the current task and submit a new task.
            self._submit_micro_batch_task(False, 0, mb)
            self._data_stream.execute()

        # Synchronize the inflight tasks.
        # Do not rotate buffers here.
        self._data_stream.synchronize()
        self._sync_inflight_operations(skip_rotate_buffers=True)


_OPTS_COORDINATOR = FlexTrainOptsCoordinator()


def get_opts_coordinator():
    """
    Get the optimizer coordinator.

    Returns:
        FlexTrainOptsCoordinator: The optimizer coordinator.
    """
    return _OPTS_COORDINATOR
