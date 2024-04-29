import torch


def print_mem():
    stats = {}
    for k, v in torch.cuda.memory_stats().items():
        if k.startswith("active_bytes.all"):
            stats[k] = v
    print(stats)


class Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, huge_input):
        ctx.save_for_backward(huge_input)
        return huge_input

    @staticmethod
    def backward(ctx, grad_output):
        huge_input, = ctx.saved_tensors
        return grad_output + huge_input

print_mem()
a = torch.randn(1024, 1024, requires_grad=False, device='cuda')

print_mem()
b = Function.apply(a)
a = torch.empty(0, device='cuda')
print_mem()
