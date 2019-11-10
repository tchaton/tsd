import torch
import tsd.scatter_cpu

if torch.cuda.is_available():
    import tsd.scatter_cuda


def get_func(name, tensor):
    if tensor.is_cuda:
        module = tsd.scatter_cuda
    else:
        module = tsd.scatter_cpu
    return getattr(module, name)
