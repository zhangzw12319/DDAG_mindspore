import os
import numpy as np
import os.path as osp
import mindspore.dataset as ds

def GenIdx(train_color_label, train_thermal_label):
    """
    Generate
    """
    color_pos = []
    unique_label_color = np.unique(train_color_label)
    for i in range(len(unique_label_color)):
        tmp_pos = [k for k,v in enumerate(train_color_label) if v==unique_label_color[i]]
        color_pos.append(tmp_pos)

    thermal_pos = []
    unique_label_thermal = np.unique(train_thermal_label)
    for i in range(len(unique_label_thermal)):
        tmp_pos = [k for k,v in enumerate(train_thermal_label) if v==unique_label_thermal[i]]
        thermal_pos.append(tmp_pos)
    
    return color_pos, thermal_pos


class IdentitySampler(ds.Sampler):
    """Sample person identities evenly in each batch.
        Args:
            train_color_label, train_thermal_label: labels of two modalities
            color_pos, thermal_pos: positions of each identity
            batchSize: batch size
    """

    def __init__(self, train_color_label, train_thermal_label, color_pos, thermal_pos, num_pos, batchSize, epoch):        
        super(IdentitySampler, self).__init__()
        uni_label = np.unique(train_color_label)
        self.n_classes = len(uni_label)
        N = np.maximum(len(train_color_label), len(train_thermal_label)) 
        for j in range(int(N/(batchSize*num_pos))+1):
            batch_idx = np.random.choice(uni_label, batchSize, replace = False)  
            for i in range(batchSize):
                sample_color  = np.random.choice(color_pos[batch_idx[i]], num_pos)
                sample_thermal = np.random.choice(thermal_pos[batch_idx[i]], num_pos)
                
                if j ==0 and i==0:
                    index1= sample_color
                    index2= sample_thermal
                else:
                    index1 = np.hstack((index1, sample_color))
                    index2 = np.hstack((index2, sample_thermal))
        
        self.index1 = index1
        self.index2 = index2
        self.N  = N
        self.num_samples = N
        
    def __iter__(self):
        # return iter(np.arange(len(self.index1))) 
        for i in range(len(self.index1)):
            yield i
        
    def __len__(self):
        return self.N

class AverageMeter(object):
    """Computers and stores the average & current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val,n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
