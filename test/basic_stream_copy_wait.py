import torch
import torch.cuda

stream = torch.cuda.Stream()


for i in range(100):
    a = torch.randn(1024 ** 2, device='cpu')
    b = torch.randn(a.numel(), device='cuda')
    c = torch.empty(2 * a.numel(), device='cuda')
    d = torch.concat([a, b.cpu()], dim=0)

    torch.cuda.synchronize()

    with torch.cuda.stream(stream):
        c1, c2 = torch.split(c, a.numel())
        c1.copy_(a, non_blocking=True)
        c2.copy_(b, non_blocking=True)

    torch.cuda.current_stream().wait_stream(stream)

    assert torch.allclose(c.cpu(), d)

print(c)
print(d)
