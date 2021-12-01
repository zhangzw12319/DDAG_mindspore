import mindspore as ms
import mindspore.dataset as ds
import numpy as np
import os as os
from PIL import Image, ImageChops

# minist_dataset = ds.MnistDataset()
from IPython import embed

class SYSUDatasetGenerator():
    """
    """
    def __init__(self, data_dir, colorIndex=None, thermalIndex=None, ifDebug=False):

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
            
        print("Color Image Size:{}".format(len(self.train_color_image)))
        print("Color Label Size:{}".format(len(self.train_color_label)))

        self.cIndex = colorIndex
        self.tIndex = thermalIndex

    def __next__(self):
        pass

    def __getitem__(self, index):

        img1, target1 = self.train_color_image[self.cIndex[index]], self.train_color_label[self.cIndex[index]]
        img2, target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]

        return (img1, img2, target1, target2)

    def __len__(self):
        # __len__ function will be called in ds.GeneratorDataset()
        # print("len_cIndex", self.cIndex.shape)
        return len(self.cIndex)


def load_data(input_data_path):
    """
    loading data
    """
    with open(input_data_path, encoding="utf-8"):
        with open(input_data_path, 'rt', encoding="utf-8") as path_str:
            data_file_list = path_str.read().splitlines()
            # Get full list of image and labels
            file_image = [s.split(' ')[0] for s in data_file_list]
            file_label = [int(s.split(' ')[1]) for s in data_file_list]

    return file_image, file_label


class RegDBDatasetGenerator():
    """
    Data Generator for custom DataLoader in mindspore, for RegDB dataset
    """
    def __init__(self, data_dir, trial, colorIndex=None, thermalIndex=None):
        # Load training images (path) and labels
        train_color_list = os.path.join(data_dir, f'idx/train_visible_{trial}' + '.txt')
        train_thermal_list = os.path.join(data_dir, f'idx/train_thermal_{trial}' + '.txt')

        color_img_file, train_color_label = load_data(train_color_list)
        thermal_img_file, train_thermal_label = load_data(train_thermal_list)

        train_color_image = []
        for _, file_ in enumerate(color_img_file):
            img = Image.open(os.path.join(data_dir, file_))
            img = img.resize((144, 288), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_color_image.append(pix_array)
        train_color_image = np.array(train_color_image)

        train_thermal_image = []
        for _, file_ in enumerate(thermal_img_file):
            img = Image.open(os.path.join(data_dir, file_))
            img = img.resize((144, 288), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_thermal_image.append(pix_array)
        train_thermal_image = np.array(train_thermal_image)

        # BGR to RGB
        self.train_color_image = train_color_image
        self.train_color_label = train_color_label

        # BGR to RGB
        self.train_thermal_image = train_thermal_image
        self.train_thermal_label = train_thermal_label

        self.cIndex = colorIndex
        self.tIndex = thermalIndex

    def __getitem__(self, index):

        img1, target1 = self.train_color_image[self.cIndex[index]],\
            self.train_color_label[self.cIndex[index]]
        img2, target2 = self.train_thermal_image[self.tIndex[index]],\
            self.train_thermal_label[self.tIndex[index]]

        return (img1, img2, target1, target2)

    def __len__(self):
        return len(self.train_color_label)


class TestData():
    def __init__(self, test_img_file, test_label, img_size=(144,288)):

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
        img1, target1 = self.test_image[index], self.test_label[index]

        return (img1, target1)

    def __len__(self):
        return len(self.test_image)