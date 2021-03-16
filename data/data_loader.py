import mindspore.dataset as ds
import numpy as np
import os as os
from PIL import Image, ImageChops

# minist_dataset = ds.MnistDataset()

class SYSUDatasetGenerator():
    """
    """
    def __init__(self, data_dir="./data/SYSU-MM01/", colorIndex=None, thermalIndex=None, ifDebug=False):

        # Load training images (path) and labels
        if ifDebug:   
            self.train_color_image = np.load(os.path.join(data_dir, 'demo_train_rgb_resized_img.npy'))
            self.train_color_label = np.load(os.path.join(data_dir, 'demo_train_rgb_resized_label.npy'))

            self.train_thermal_image = np.load(os.path.join(data_dir, 'demo_train_ir_resized_img.npy'))
            self.train_thermal_label = np.load(os.path.join(data_dir,  'demo_train_ir_resized_label.npy'))
        else:
            self.train_color_image = np.load(os.path.join(data_dir, 'train_rgb_resized_img.npy'))
            self.train_color_label = np.load(os.path.join(data_dir, 'train_rgb_resized_label.npy'))

            self.train_thermal_image = np.load(os.path.join(data_dir, 'train_ir_resized_img.npy'))
            self.train_thermal_label = np.load(os.path.join(data_dir,  'train_ir_resized_label.npy'))

        print("Color Igame Size:{}".format(len(self.train_color_image)))
        print("Color Label Size:{}".format(len(self.train_color_label)))

        self.cIndex = colorIndex
        self.tIndex = thermalIndex

    def __next__(self):
        pass

    def __getitem__(self, index):
        # TODO: 这里要配合samplers输出的更改而更改
        # print(index)
        # print("self.cIndex is ",self.cIndex[index] )
        # print("self.tIndex is ",self.tIndex[index] )
        img1,  target1 = self.train_color_image[self.cIndex[index]],  self.train_color_label[self.cIndex[index]]
        img2,  target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]
        # print("img1 is:", img1)
        # print("target1 is:", target1)
        # print("img2 is:", img2)
        # print("target2 is:", target2)

        return (img1, img2, target1, target2)

    def __len__(self):
        # original version
        return len(self.train_color_label)


class TestData():
    def __init__(self, test_img_file, test_label, img_size = (144,288)):

        test_image = []
        for i in range(len(test_img_file)):
            img = Image.open(test_img_file[i])
            img = img.resize((img_size[0], img_size[1]), Image.ANTIALIAS)
            pix_array = np.array(img)
            test_image.append(pix_array)
        test_image = np.array(test_image)
        self.test_image = test_image
        self.test_label = test_label

    def __getitem__(self, index):
        img1,  target1 = self.test_image[index],  self.test_label[index]
        return img1, target1

    def __len__(self):
        return len(self.test_image)