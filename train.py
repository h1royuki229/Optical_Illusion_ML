import torch
from torch.utils.data import DataLoader
import numpy as np

from color_net import ColorNet
from util import *
from hyperparam import *



def train(net, device, train_img, train_label, save_model_path):

    optimizer = torch.optim.Adam(params=net.parameters(), lr=0.001)
    loss_func = torch.nn.CrossEntropyLoss()

    # (サイズk チャネル数, width, height)=(枚数, RGB(3), 28, 28)
    train_img = train_img.permute(0, 3, 2, 1)
    dataset = torch.utils.data.TensorDataset(train_img, train_label)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(EPOCHS):
        loss_list = []
        net.train()

        for i, (data, target) in enumerate(dataloader):

            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output = net(data)
            loss = loss_func(output, target.long())
            loss_list.append(loss.item())
            loss.backward()
            optimizer.step()

        print("Epoch:" + str(epoch) + ", Loss:" + str(sum(loss_list)/len(loss_list)))

    torch.save(net.state_dict(), save_model_path)