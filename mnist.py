import torch
import torch.nn.functional as f
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

def load_MNIST(batch=128, intensity=1.0):
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data',
                       train=True,
                       download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Lambda(lambda x: x * intensity)
                           # transforms.Nomalize((0.5,), (0.5,))
                       ])),
        batch_size = batch,
        shuffle = True
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data',
                       train=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Lambda(lambda x: x * intensity)
                           # transforms.Nomalize((0.5,), (0.5,))
                       ])),
        batch_size = batch,
        shuffle = True
    )

    return {'train': train_loader, 'test': test_loader}