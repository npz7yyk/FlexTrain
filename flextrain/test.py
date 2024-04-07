import torch
# from flextrain.utils import dist
# from flextrain.utils.logging import logger

# dist.init_distributed()

# t = torch.randn(16, device=dist.current_device())
# logger.info(f"Rank {dist.get_rank()} element 0 to 4: {t[:5]}")
# dist.broadcast(t, 1)

# logger.info(f"Rank {dist.get_rank()} element 0 to 4: {t[:5]}")

from flextrain.utils.memory import free_tensor, ContiguousTensorGroup
if __name__ == "__main__":
    # Create a group of tensors.
    a = torch.randint(0, 2, (2, 3))
    b = torch.randint(0, 2, (3, 4))
    c = torch.randint(0, 2, (4, 5))

    # Print the tensors.
    print(a)
    print(b)
    print(c)

    # Create a target contiguous memory.
    target = torch.empty(2 * 3 + 3 * 4 + 4 * 5, dtype=torch.int64)

    # Create a contiguous tensor group.
    group = ContiguousTensorGroup([a, b, c], target)

    # Print the tensors.
    print(a)
    print(b)
    print(c)
    print(target)

    # Free the memory of a b c.
    free_tensor(a)
    free_tensor(b)
    free_tensor(c)

    # Print the tensors.
    print(a)
    print(b)
    print(c)
    print(target)

    # Recover the view on the contiguous memory.
    group.recover_view_on(target)

    # Print the tensors.
    print(a)
    print(b)
    print(c)
    print(target)
