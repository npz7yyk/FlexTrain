import os
import torch

from constants import (
    DEFAULT_PROCESS_GROUP_TIMEOUT,
    DEFAULT_TORCH_DISTRIBUTED_BACKEND,
    DEFAULT_TORCH_DISTRIBUTED_INIT_METHOD
)
from logging_utils import rank0_info as info
from torch_utils import (
    has_all_reduce_coalesced,
    has_coalescing_manager,
    get_all_gather_function,
    get_coalescing_manager,
    get_reduce_scatter_function
)


# TODO: comments
_BACKEND = None


def is_distributed_intialized():
    return _BACKEND is not None


def init_distributed(
    dist_backend=DEFAULT_TORCH_DISTRIBUTED_BACKEND,
    timeout=DEFAULT_PROCESS_GROUP_TIMEOUT,
    init_method=DEFAULT_TORCH_DISTRIBUTED_INIT_METHOD,
    rank=-1,
    world_size=-1
):
    """
    Initialize dist backend, potentially performing MPI discovery if needed

    Arguments:
        dist_backend: Optional (str). Default is nccl.
            torch distributed backend, e.g., nccl, mpi, gloo
        timeout: Optional (timedelta). Default value equals 30 minutes.
            Timeout for operations executed against the process group.
        init_method: Optional (string). Default is “env://”.
            URL specifying how to initialize the process group.
        rank: Optional (int). Default is -1.
            The current manually specified rank. Needed by some init_methods.
        world_size: Optional (int). Default is -1.
            Desired world_size for the TCP / Shared file-system initialization.
    """
    if is_distributed_intialized():
        return

    global _BACKEND

    info(f"FlexTrain initializing TorchBackend with backend {dist_backend}")
    _BACKEND = TorchBackend(
        dist_backend=dist_backend,
        timeout=timeout,
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )


class TorchBackend(object):
    """
        A light-weight wrapper class for torch.distributed API.
        Only a subset of functions are wrapped. Once the init_process_group
        is initialized, standard torch.distributed.* can be used directly.
    """

    def __init__(
        self,
        dist_backend,
        timeout,
        init_method,
        rank,
        world_size
    ):
        self.world_size = world_size
        self.rank = rank

        self.has_all_reduce_coalesced = has_all_reduce_coalesced()
        self.has_coalescing_manager = has_coalescing_manager()
        self.all_gather_function = get_all_gather_function()
        self.reduce_scatter_function = get_reduce_scatter_function()

        # Future functionality to support ds.initialize() on a single GPU
        # The idea is to fake that dist backend is initialized even when it is
        # not so we can run on a single GPU without init_process_group
        self.single_gpu_mode = True

        if torch.distributed.is_initialized():
            return

        torch.distributed.init_process_group(
            dist_backend,
            timeout=timeout,
            init_method=init_method,
            rank=rank,
            world_size=world_size
        )

    def has_all_gather_into_tensor(self):
        return self.all_gather_function is not None

    def has_reduce_scatter_tensor(self):
        return self.reduce_scatter_function is not None

    def all_reduce(self, tensor, op=torch.distributed.ReduceOp.SUM, group=None, async_op=False):
        op = self._reduce_op(op)
        return torch.distributed.all_reduce(tensor=tensor, op=op, group=group, async_op=async_op)

    def inference_all_reduce(self, tensor, op=torch.distributed.ReduceOp.SUM, group=None, async_op=False):
        op = self._reduce_op(op)
        return torch.distributed.all_reduce(tensor=tensor, op=op, group=group, async_op=async_op)

    def all_reduce_coalesced(self, tensors, op=torch.distributed.ReduceOp.SUM, group=None, async_op=False):
        """ proxy func to torch.distributed.all_reduce_coalesced,
        which is included in PyTorch 1.13 and above
        """
        if not self.has_all_reduce_coalesced:
            raise RuntimeError(f"Current torch version does not have all_reduce_coalesced "
                               f"api (torch.__version__: {torch.__version__})")
        op = self._reduce_op(op)
        return torch.distributed.all_reduce_coalesced(tensors=tensors, op=op, group=group, async_op=async_op)

    def reduce(self, tensor, dst, op=ReduceOp.SUM, group=None, async_op=False):
        return torch.distributed.reduce(tensor=tensor, dst=dst, op=self._reduce_op(op), group=group, async_op=async_op)

    def reduce_scatter(self, output, input_list, op=ReduceOp.SUM, group=None, async_op=False):
        return torch.distributed.reduce_scatter(output=output,
                                                input_list=input_list,
                                                op=self._reduce_op(op),
                                                group=group,
                                                async_op=async_op)

    def broadcast(self, tensor, src, group=None, async_op=False):
        return torch.distributed.broadcast(tensor=tensor, src=src, group=group, async_op=async_op)

    def all_gather(self, tensor_list, tensor, group=None, async_op=False):    
        return torch.distributed.all_gather(tensor_list=tensor_list, tensor=tensor, group=group, async_op=async_op)

    def all_gather_into_tensor(self, output_tensor, input_tensor, group=None, async_op=False):
        if self.has_all_gather_into_tensor():
            return self.all_gather_function(output_tensor=output_tensor,
                                            input_tensor=input_tensor,
                                            group=group,
                                            async_op=async_op)

    def all_gather_base(self, output_tensor, input_tensor, group=None, async_op=False):
        if DS_COMM_ALL_GATHER_OFF:
            if int(os.getenv('RANK', '0')) == 0:
                utils.logger.warning("All Gather is OFF")
            return Noop()
        else:
            if self.has_allgather_base:
                return torch.distributed.distributed_c10d._all_gather_base(output_tensor=output_tensor,
                                                                           input_tensor=input_tensor,
                                                                           group=group,
                                                                           async_op=async_op)
            else:
                utils.logger.warning("unable to find torch.distributed._all_gather_base. will fall back to "
                                     "torch.distributed.reduce_scatter which will result in suboptimal performance. "
                                     "please consider upgrading your pytorch installation.")
                pass

    def all_gather_coalesced(self, output_tensors, input_tensors, group=None, async_op=False):
        """"""
        assert len(output_tensors) == len(input_tensors), ""
        if hasattr(torch.distributed.distributed_c10d, '_all_gather_base_coalesced'):
            # customized PyTorch
            return torch.distributed.distributed_c10d._all_gather_base_coalesced(output_tensors,
                                                                                 input_tensors,
                                                                                 group=group,
                                                                                 async_op=async_op)
        elif has_coalescing_manager():
            reqs = []
            with get_coalescing_manager(group, input_tensors[0].device, reqs, async_op):
                for output, input in zip(output_tensors, input_tensors):
                    handle = torch.distributed.distributed_c10d.all_gather_into_tensor(output,
                                                                                       input,
                                                                                       group=group,
                                                                                       async_op=True)
                    reqs.append(handle)
            if async_op:
                return reqs[-1]
            else:
                reqs[-1].wait()

    def reduce_scatter_tensor(self, output_tensor, input_tensor, op=ReduceOp.SUM, group=None, async_op=False):
        if self.has_reduce_scatter_tensor():
            return self.reduce_scatter_function(output_tensor,
                                                input_tensor,
                                                op=self._reduce_op(op),
                                                group=group,
                                                async_op=async_op)
        else:
            utils.logger.warning("unable to find torch.distributed.reduce_scatter_tensor. will fall back to "
                                 "torch.distributed.reduce_scatter which will result in suboptimal performance. "
                                 "please consider upgrading your pytorch installation.")
            pass

    def all_to_all_single(self,
                          output,
                          input,
                          output_split_sizes=None,
                          input_split_sizes=None,
                          group=None,
                          async_op=False):
        return torch.distributed.all_to_all_single(output=output,
                                                   input=input,
                                                   output_split_sizes=output_split_sizes,
                                                   input_split_sizes=input_split_sizes,
                                                   group=group,
                                                   async_op=async_op)

    def all_to_all(self, output_tensor_list, input_tensor_list, group=None, async_op=False):
        return torch.distributed.all_to_all(output_tensor_list, input_tensor_list, group=group, async_op=async_op)

    def send(self, tensor, dst, group=None, tag=0):
        return torch.distributed.send(tensor=tensor, dst=dst, group=group, tag=tag)

    def recv(self, tensor, src=None, group=None, tag=0):
        return torch.distributed.recv(tensor=tensor, src=src, group=group, tag=tag)

    def isend(self, tensor, dst, group=None, tag=0):
        return torch.distributed.isend(tensor=tensor, dst=dst, group=group, tag=tag)

    def irecv(self, tensor, src=None, group=None, tag=0):
        return torch.distributed.irecv(tensor=tensor, src=src, group=group, tag=tag)

    def gather(self, tensor, gather_list=None, dst=0, group=None, async_op=False):
        return torch.distributed.gather(tensor=tensor,
                                        gather_list=gather_list,
                                        dst=dst,
                                        group=group,
                                        async_op=async_op)

    def scatter(self, tensor, scatter_list=None, src=0, group=None, async_op=False):
        return torch.distributed.scatter(tensor=tensor,
                                         scatter_list=scatter_list,
                                         src=src,
                                         group=group,
                                         async_op=async_op)

    def barrier(self, group=torch.distributed.GroupMember.WORLD, async_op=False, device_ids=None):
        if group is None:
            group = torch.distributed.GroupMember.WORLD
        return torch.distributed.barrier(group=group, async_op=async_op, device_ids=device_ids)

    def monitored_barrier(self, group=torch.distributed.GroupMember.WORLD, timeout=None, wait_all_ranks=False):
        if group is None:
            group = torch.distributed.GroupMember.WORLD
        return torch.distributed.monitored_barrier(group=group, timeout=timeout, wait_all_ranks=wait_all_ranks)

    def get_rank(self, group=None):
        return torch.distributed.get_rank(group=group)

    def get_world_size(self, group=None):
        return torch.distributed.get_world_size(group=group)

    def is_initialized(self):
        return torch.distributed.is_initialized()

    def get_backend(self, group=None):
        return torch.distributed.get_backend(group=group)

    def new_group(self, ranks):
        return torch.distributed.new_group(ranks)

    def get_global_rank(self, group, group_rank):
        if hasattr(torch.distributed.distributed_c10d, "get_global_rank"):
            from torch.distributed.distributed_c10d import get_global_rank as _get_global_rank
        else:
            from torch.distributed.distributed_c10d import _get_global_rank
        return _get_global_rank(group, group_rank)

    def get_world_group(self):
        return torch.distributed.group.WORLD

    def destroy_process_group(self, group=None):
        return torch.distributed.destroy_process_group(group=group)

    def _reduce_op(self, op):
        '''
            Helper function. If the op provided is not a torch.dist.ReduceOp, convert it and return
        '''
        if not isinstance(op, torch.distributed.ReduceOp):
            if op == ReduceOp.SUM:
                op = torch.distributed.ReduceOp.SUM
            elif op == ReduceOp.PRODUCT:
                op = torch.distributed.ReduceOp.PRODUCT
            elif op == ReduceOp.AVG:
                op = torch.distributed.ReduceOp.AVG
            elif op == ReduceOp.MIN:
                op = torch.distributed.ReduceOp.MIN
            elif op == ReduceOp.MAX:
                op = torch.distributed.ReduceOp.MAX
            elif op == ReduceOp.BAND:
                op = torch.distributed.ReduceOp.BAND
            elif op == ReduceOp.BOR:
                op = torch.distributed.ReduceOp.BOR
            elif op == ReduceOp.BXOR:
                op = torch.distributed.ReduceOp.BXOR
        return op

