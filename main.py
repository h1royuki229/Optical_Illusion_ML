from color_net import ColorNet
from colored_mnist import ColoredMnist
from train import train
from test import test
from util import *


def main(create_data):

    if create_data:
        color_mnist = ColoredMnist()
    else:
        filename = "colored_mnist_data.pickle"
        color_mnist = load_pickle(filename)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = ColorNet(len(color_mnist.colorlist))
    net.to(device)


    saved_model = "./model.pth"

    print("########## Train ##########")
    train(net, device, color_mnist.train_img, color_mnist.train_color_label, saved_model)


    ########## Trainを行わずにTestのみを行う場合 ##########
    # net = ColorNet(len(color_mnist.colorlist))
    # net.load_state_dict(torch.load(saved_model))
    # net.to(device)


    print("########## Normal Test ##########")
    test(net, device, color_mnist.normal_test_img, color_mnist.test_color_label, saved_model)

    print("########## Illusion Test ##########")
    test(net, device, color_mnist.ill_test_img, color_mnist.ill_color_label, saved_model)



if __name__ == '__main__':
    create_data = False
    main(create_data)
