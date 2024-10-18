# Parts of the code here are adapted from DeepSpeed
#
# Repository: https://github.com/microsoft/DeepSpeed
# File: deepspeed/ops/op_builder/cpu_adam.py
# Commit: 0d9cfa0
# License: Apache-2.0

from .builder import CPUOpBuilder


class CPUAdamBuilder(CPUOpBuilder):
    BUILD_VAR = "FT_BUILD_CPU_ADAM"
    NAME = "cpu_adam"

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f'flextrain.ops.adam.{self.NAME}_op'

    def sources(self):
        return ['csrc/adam/cpu_adam.cpp', 'csrc/adam/cpu_adam_impl.cpp']

    def libraries_args(self):
        args = super().libraries_args()
        return args

    def include_paths(self):
        return ['csrc/includes']
