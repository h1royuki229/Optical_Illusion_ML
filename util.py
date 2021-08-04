import torch

def use_gpu(e):
    if torch.cuda.is_available():
        return e.cuda()
    return e


