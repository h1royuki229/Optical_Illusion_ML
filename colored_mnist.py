import torch
import numpy as np
from matplotlib import colors
import os
from PIL import Image

from util import *


class ColoredMnist():

    def __init__(self):
        dataset = np.load("colored_mnist_gen/mnist_10color_jitter_var_0.020.npy", encoding="latin1", allow_pickle=True).item()

        self.colorlist = ["r", "g", "b", "c", "m", "y", "k", "w"]

        test_img = torch.from_numpy(dataset["test_image"].astype(np.float32)).clone()
        self.test_label = torch.from_numpy(dataset["test_label"].astype(np.float32)).clone()
        train_img = torch.from_numpy(dataset["train_image"].astype(np.float32)).clone()
        self.train_label = torch.from_numpy(dataset["train_label"].astype(np.float32)).clone()


        self.train_img, self.train_color_label, self.train_one_hot = self.recolor_mnist(train_img)
        self.normal_test_img, self.test_color_label, self.test_one_hot = self.recolor_mnist(test_img)
        self.ill_test_img, self.ill_color_label, self.ill_one_hot = self.illusion_mnist(test_img)

        dump_pickle("colored_mnist_data.pickle", self)


    def one_img_show(self, img):
        pil_img = Image.fromarray(np.uint8(img*255))
        # pil_img.show()
        return pil_img


    def recolor_mnist(self, origin_img):
        img = origin_img
        # one-hot表記にする
        color_label = torch.zeros(origin_img.shape[0])
        color_one_hot = torch.zeros(origin_img.shape[0], len(self.colorlist))

        for i in range(img.shape[0]):
            shape = img[i].shape
            img_array = img[i].view(-1, 3)
            i_color = np.random.choice(len(self.colorlist), size=2, replace=False)
            color_num = i_color[0]
            background_num = i_color[1]
            digit_rgb = torch.tensor(colors.to_rgb(self.colorlist[color_num]))
            back_rgb = torch.tensor(colors.to_rgb(self.colorlist[background_num]))

            for j in range(img_array.shape[0]):
                if torch.sum(img_array[j]) != 0:
                    img_array[j] = digit_rgb
                else:
                    img_array[j] = back_rgb

            img[i] = img_array.view(shape)

            color_label[i] = color_num
            one_hot = torch.zeros(len(self.colorlist))
            one_hot[color_num] = 1
            color_one_hot[i] = one_hot

        return img, color_label, color_one_hot


    def illusion_mnist(self, origin_img):
        img = origin_img
        color_label = torch.zeros(origin_img.shape[0])
        color_one_hot = torch.zeros(origin_img.shape[0], len(self.colorlist))

        for i in range(img.shape[0]):
            shape = img[i].shape
            img_array = img[i].view(-1, 3)
            i_color = np.random.choice(len(self.colorlist), size=3, replace=False)
            color_num = i_color[0]
            background_num = i_color[1]
            illusion_num = i_color[2]
            illusion_width = 2
            digit_rgb = torch.tensor(colors.to_rgb(self.colorlist[color_num]))
            back_rgb = torch.tensor(colors.to_rgb(self.colorlist[background_num]))
            ill_rgb = torch.tensor(colors.to_rgb(self.colorlist[illusion_num]))
            # 縦線(長さ, 幅, rgb)
            ill_rgb = ill_rgb.repeat(shape[1], illusion_width, 1)

            for j in range(img_array.shape[0]):
                if torch.sum(img_array[j]) != 0:
                    img_array[j] = digit_rgb
                else:
                    img_array[j] = back_rgb

            img[i] = img_array.view(shape)

            # 縦線を入れる(左右対称に入るように1から)
            for j in range(1, shape[1], illusion_width*2):
                if j <= shape[1]-2:
                    img[i][:,j:j+2,:] = ill_rgb


            color_label[i] = color_num
            one_hot = torch.zeros(len(self.colorlist))
            one_hot[color_num] = 1
            color_one_hot[i] = one_hot

        return img, color_label, color_one_hot


    def imgs_show(self, imgs):
        # 幅，高さ，文字数
        chr_w = 28
        chr_h = 28
        num = 16

        # MNISTの文字をPILで１枚の画像に描画する
        canvas = Image.new('RGB', (int(chr_w * num/2), int(chr_h * num/2)), (255, 255, 255))

        i = 0
        for y in range(int(num/2)):
            for x in range(int(num/2)):
                canvas.paste(self.one_img_show(imgs[i]), (chr_w*x, chr_h*y))
                i = i + 1

        # canvas.show()

        # 表示した画像をJPEGとして保存
        canvas.save('colored_mnist.jpg', 'JPEG', quality=100, optimize=True)
