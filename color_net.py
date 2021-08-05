import torch
import torch.nn as nn
import torch.nn.functional as f


class ColorNet(torch.nn.Module):

    def __init__(self, color_num):
        super(ColorNet, self).__init__()

        # Conv2d --> (in_channels, out_channels, kerel_size, padding)
        self.conv1 = nn.Conv2d(3, 8, 3, 1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(8, 4, 3, 1)
        self.pool2 = nn.MaxPool2d(2)
        self.fc = nn.Linear(100, color_num)
        self.relu = nn.ReLU()


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
        x = self.fc(x)


        return x