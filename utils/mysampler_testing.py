import mindspore.dataset as ds
import numpy as np

class GeneratorMD():
    def __init__(self):
        self.raw_data = np.array([0,1,2,3,4,5,6,7,8,9])
        self.raw_label = np.array([9,8,7,6,5,4,3,2,1,0])

    def __getitem__(self, index):
        return self.raw_data[index], self.raw_label[index]

    def __len__(self):
        return len(self.raw_data)

class MySampler(ds.Sampler):
    def __iter__(self):
        for i in range(0, 10, 1):
            yield i

dataset = ds.GeneratorDataset(GeneratorMD(), ["col1", "col2"], sampler=MySampler())


for data in dataset.create_dict_iterator():
    print("Image:", data['col1'], ", Label:", data['col2'])
    