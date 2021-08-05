import torch
import torch.nn as nn
import torch.nn.functional as f


class ColorNet(torch.nn.Module):

    def __init__(self, color_num):
        super(ColorNet, self).__init__()

        # Conv2d --> (in_channels, out_channels, kerel_size, padding)
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 16, 3, 1)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(16, 8, 3, 1)
        self.pool3 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(400, 100)
        self.fc2 = nn.Linear(100, color_num)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()


    def forward(self, x):
        batch_size = x.shape[0]

        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)

        # Linearに入力するために1次元にする
        x = x.view(batch_size, -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x