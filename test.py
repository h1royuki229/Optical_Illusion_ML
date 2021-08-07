import torch
import  torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from color_net import ColorNet
from util import *
from hyperparam import *



def test(net, device, test_img, test_label, save_model_path):

    test_img = test_img.permute(0,3,2,1)
    test_size = test_img.shape[0]
    dataset = torch.utils.data.TensorDataset(test_img, test_label)
    dataloader = DataLoader(dataset, batch_size=1)

    net.eval()
    test_loss = []
    correct = 0

    calc_softmax = nn.Softmax()

    with torch.no_grad():
        for i, (data, target) in enumerate(dataloader):

            data = data.to(device)
            target = target.to(device)

            output = net(data)
            output = calc_softmax(output)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    print((correct/test_size)*100)
