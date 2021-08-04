import torch
import numpy as np
from matplotlib import colors
import os
from PIL import Image

class ColoredMnist():
    def __init__(self):
        dataset = np.load("colored_mnist_gen/mnist_10color_jitter_var_0.020.npy", encoding="latin1", allow_pickle=True).item()

        self.colorlist = ["r", "g", "b", "c", "m", "y", "k", "w"]
        # torch.tensor(colors.to_rgb("y"))

        test_img = torch.from_numpy(dataset["test_image"].astype(np.float32)).clone()
        self.test_label = torch.from_numpy(dataset["test_label"].astype(np.float32)).clone()
        train_img = torch.from_numpy(dataset["train_image"].astype(np.float32)).clone()
        self.train_label = torch.from_numpy(dataset["train_label"].astype(np.float32)).clone()


        self.recolor_test_img, self.test_color_label = self.recolor_mnist(test_img)
        self.recolor_train_img, self.train_color_label = self.recolor_mnist(train_img)

        self.dump_pickle("colored_mnist_data.pickle", self)


    def dump_pickle(self, file, obj):
        with open(file, mode="wb") as f:
            pickle.dump(obj, f)


    def img_show(self, img):
        pil_img = Image.fromarray(np.uint8(img*255))
        pil_img.show()


    def recolor_mnist(self, origin_img):
        img = origin_img
        color_label = torch.zeros(origin_img.shape[0])

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

        return img, color_label
