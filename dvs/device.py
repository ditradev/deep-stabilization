import torch


def get_device(use_cuda: bool = True) -> torch.device:
    """Return the torch.device to use based on configuration and availability."""

    if use_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
