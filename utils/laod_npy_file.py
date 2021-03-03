import numpy as np
from PIL import Image, ImageChops

if __name__ == "__main__":
    data_dir = "./data/SYSU-MM01/"

    train_color_image = np.load(data_dir + "train_rgb_resized_label.npy")

    print(train_color_label)