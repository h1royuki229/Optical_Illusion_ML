import torch
import pickle

def cross_entropy(output, target):
    # average(-Σp(x)log(q(x)))
    loss = torch.mean(torch.mean(target * torch.log(output), dim=1) * -1)
    return loss


def dump_pickle(file, obj):
    with open(file, mode="wb") as f:
        pickle.dump(obj, f)


def load_pickle(file_name):
    with open(file_name, mode="rb") as f:
        return pickle.load(f)