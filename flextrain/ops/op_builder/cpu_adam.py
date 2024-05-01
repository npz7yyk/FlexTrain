# Parts of the code here are adapted from DeepSpeed
#
# Repository: https://github.com/microsoft/DeepSpeed
# File: deepspeed/ops/op_builder/cpu_adam.py
# Commit: 0d9cfa0
# License: Apache-2.0

import os
from .builder import TorchCPUOpBuilder


class CPUAdamBuilder(TorchCPUOpBuilder):
    BUILD_VAR = "FT_BUILD_CPU_ADAM"
    NAME = "cpu_adam"

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f'flextrain.ops.adam.{self.NAME}_op'

    def sources(self):
        if self.build_for_cpu:
            return ['csrc/adam/cpu_adam.cpp', 'csrc/adam/cpu_adam_impl.cpp']
        else:
            return [
                'csrc/adam/cpu_adam.cpp', 'csrc/adam/cpu_adam_impl.cpp',
                'csrc/common/custom_cuda_kernel.cu'
            ]

    def libraries_args(self):
        args = super().libraries_args()
        if self.build_for_cpu:
            return args

        if not self.is_rocm_pytorch():
            args += ['curand']

        return args

    def include_paths(self):
        import torch
        if self.build_for_cpu:
            CUDA_INCLUDE = []
        elif not self.is_rocm_pytorch():
            CUDA_INCLUDE = [
                os.path.join(torch.utils.cpp_extension.CUDA_HOME, "include")]
        else:
            CUDA_INCLUDE = []
        return ['csrc/includes'] + CUDA_INCLUDE
