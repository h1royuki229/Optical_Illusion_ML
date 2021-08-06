import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt


from color_net import ColorNet
from colored_mnist import ColoredMnist
from util import *


def main():

    # tmp = ColoredMnist()
    filename = "colored_mnist_data.pickle"
    color_mnist = load_pickle(filename)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    epochs = 20

    history = {
        'train_loss': [],
        'test_acc': [],
        'ill_test_acc': []
    }

    net = ColorNet(len(color_mnist.colorlist))
    net.to(device)

    optimizer = torch.optim.Adam(params=net.parameters(), lr=0.001)
    loss_func = torch.nn.CrossEntropyLoss()

    ############################ Train Model ############################
    # (サイズk チャネル数, width, height)=(枚数, RGB(3), 28, 28)
    train_img = color_mnist.train_img.permute(0, 3, 2, 1)
    dataset = torch.utils.data.TensorDataset(train_img, color_mnist.train_color_label)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    for epoch in range(epochs):
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

        history['train_loss'].append(sum(loss_list)/len(loss_list))

    ############################ Test Model(use train_data) ############################
    test_img = color_mnist.train_img.permute(0, 3, 2, 1)
    test_size = test_img.shape[0]
    dataset = torch.utils.data.TensorDataset(test_img, color_mnist.train_color_label)
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

    history['test_acc'].append((correct/test_size) * 100)

    print("Train Accuracy")
    print((correct/test_size) * 100)



    ############################ Test Model ############################
    test_img = color_mnist.normal_test_img.permute(0, 3, 2, 1)
    test_size = test_img.shape[0]
    dataset = torch.utils.data.TensorDataset(test_img, color_mnist.test_color_label)
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

    history['test_acc'].append((correct/test_size) * 100)


    ############################ Illusion Test Model ############################
    test_img = color_mnist.ill_test_img.permute(0, 3, 2, 1)
    test_size = test_img.shape[0]
    dataset = torch.utils.data.TensorDataset(test_img, color_mnist.ill_color_label)
    dataloader = DataLoader(dataset, batch_size=1)

    net.eval()
    ill_test_loss = []
    ill_correct = 0
    wrong_index = []
    wrong_output = []

    with torch.no_grad():
        for i, (data, target) in enumerate(dataloader):

            data = data.to(device)
            target = target.to(device)

            output = net(data)
            output = calc_softmax(output)
            pred = output.argmax(dim=1, keepdim=True)
            ill_correct += pred.eq(target.view_as(pred)).sum().item()

            # 認識が間違っていた場合
            if pred.eq(target.view_as(pred)).sum().item() == 0:
                wrong_index.append(i)
                wrong_output.append(pred)


    color_mnist.one_img_show(color_mnist.ill_test_img[wrong_index[0]])

    history['ill_test_acc'].append((ill_correct/test_size) * 100)


    # 結果の出力
    print(history)

if __name__ == '__main__':
    main()
