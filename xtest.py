from flextrain.ops.adam import FlexTrainCPUAdam
import torch
a = FlexTrainCPUAdam([torch.tensor([1.0, 2.0, 3.0], requires_grad=True), torch.tensor([4.0, 5.0, 6.0], requires_grad=True)])
print(a.param_groups)
