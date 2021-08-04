import torch
import torch.nn.functional as f
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from SimpleNet import SimpleNet
from util import use_gpu
from mnist import load_MNIST


def use_gpu(e):
    if torch.cuda.is_available():
        return e.cuda()
    return e

def main():

    epochs = 20

    history = {
        'train_loss': [],
        'test_loss': [],
        'test_acc': [],
    }

    net: torch.nn.Module = SimpleNet()
    net = use_gpu(net)
    loaders = load_MNIST()

    optimizer = torch.optim.Adam(params=net.parameters(), lr=0.001)

    for epoch in range(epochs):
        loss = None

        # 学習開始・再開
        net.train(True)

        for i, (data, target) in enumerate(loaders['train']):

            data = use_gpu(data)
            target = use_gpu(target)

            # 全結合のみのネットワークでは入力を1次元にする
            data = data.view(-1, 28*28)

            optimizer.zero_grad()
            output = net(data)
            loss = f.nll_loss(output, target)
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print('Training log: {} epoch({} / 60000 train. data). Loss: {}'.format(epoch+1,
                                                                                        (i+1)*128,
                                                                                        loss.item()
                                                                                        )
                )

        history['train_loss'].append(loss)

        # 学習停止
        net.eval()
        # net.train(False)

        test_loss = 0
        correct = 0

        with torch.no_grad(): # テスト部分では勾配は使わない
            for data, target in loaders['test']:

                data = use_gpu(data)
                target = use_gpu(target)

                data = data.view(-1, 28*28)
                output = net(data)
                test_loss += f.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= 10000

        print('Test loss (avg): {}, Accuracy: {}'.format(test_loss,
                                                         correct / 10000
                                                        )
        )

        history['test_loss'].append(test_loss)
        history['test_acc'].append(correct / 10000)


    # 結果の出力と描画
    print(history)
    plt.figure()
    plt.plot(range(1, epoch+2), history['train_loss'], label='train_loss')
    plt.plot(range(1, epoch+2), history['test_loss'], label='test_loss')
    plt.xlabel('epoch')
    plt.legend()
    # plt.savefig('loss.png')
    plt.show()

    plt.figure()
    plt.plot(range(1, epoch+2), history['test_acc'])
    plt.title('test accuracy')
    plt.xlabel('epoch')
    # plt.savefig('test_acc.png')
    plt.show()


if __name__ == '__main__':
    main()
