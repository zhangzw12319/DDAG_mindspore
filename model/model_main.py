from unicodedata import normalize
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as P
import mindspore.common.initializer as init
from mindspore.common.initializer import Initializer, _assignment, random_normal
from model.resnet import *
from model.attention import IWPA

import os
import numpy as np
import psutil
from IPython import embed

def show_memory_info(hint=""):
    pid = os.getpid()

    p = psutil.Process(pid)
    info = p.memory_full_info()
    memory = info.uss/1024./1024
    print(f"{hint} memory used: {memory} MB ")

class Normal_with_mean(Initializer):
    """
    Initialize a normal array, and obtain values N(0, sigma) from the uniform distribution
    to fill the input tensor.

    Args:
        sigma (float): The sigma of the array. Default: 0.01.

    Returns:
        Array, normal array.
    """
    def __init__(self, mu=0, sigma=0.01):
        super(Normal_with_mean, self).__init__(sigma=sigma)
        self.mu = mu
        self.sigma = sigma

    def _initialize(self, arr):
        seed, seed2 = self.seed
        normalize
        output_tensor = ms.Tensor(np.zeros(arr.shape, dtype=np.float32) + np.ones(arr.shape, dtype=np.float32) * self.mu)
        random_normal(arr.shape, seed, seed2, output_tensor)
        output_data = output_tensor.asnumpy()
        output_data *= self.sigma
        _assignment(arr, output_data)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.set_data(init.initializer(init.HeNormal(negative_slope=0, mode='fan_in'), m.weight.shape, m.weight.dtype))
    elif classname.find('Linear') != -1:
        m.weight.set_data(init.initializer(init.HeNormal(negative_slope=0, mode='fan_out'), m.weight.shape, m.weight.dtype))
        m.bias.set_data(init.initializer(init.Zero(), m.bias.shape, m.bias.dtype))
    elif classname.find('BatchNorm1d') != -1:
        m.gamma.set_data(init.initializer(Normal_with_mean(mu=1, sigma=0.01), m.gamma.shape, m.gamma.dtype))
        m.beta.set_data(init.initializer(init.Zero(), m.beta.shape, m.beta.dtype))

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.gamma.set_data(init.initializer(init.Normal(sigma=0.001), m.gamma.shape, m.gamma.dtype))
        if m.bias:
            m.bias.set_data(init.initializer(init.Zero(), m.bias.shape, m.bias.dtype))

class Normalize(nn.Cell):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power
        self.pow = P.Pow()
        self.sum = P.ReduceSum(keep_dims=True)
        self.div = P.Div()

    def construct(self, x):
        norm = self.pow(x, self.power)
        norm = self.sum(norm, 1)
        norm = self.pow(norm, 1. / self.power)
        out = self.div(x, norm)
        return out


class visible_module(nn.Cell):
    def __init__(self, arch="resnet50", pretrain=""):
        super(visible_module, self).__init__()

        # self.visible = resnet50(pretrain=pretrain)
        self.visible = resnet50_specific(pretrain=pretrain)
        
    def construct(self, x):
        # x = self.visible.conv1(x)
        # x = self.visible.bn1(x)
        # x = self.visible.relu(x)
        # x = self.visible.maxpool(x)

        x = self.visible(x)

        return x


class thermal_module(nn.Cell):
    def __init__(self, arch="resnet50", pretrain=""):
        super(thermal_module, self).__init__()

        # self.thermal = resnet50(pretrain=pretrain)
        self.thermal = resnet50_specific(pretrain=pretrain)

    def construct(self, x):
        # x = self.thermal.conv1(x)
        # x = self.thermal.bn1(x)
        # x = self.thermal.relu(x)
        # x = self.thermal.maxpool(x)

        x = self.thermal(x)

        return x


class base_resnet(nn.Cell):
    def __init__(self, arch="resnet50", pretrain=""):
        super(base_resnet, self).__init__()

        # self.base = resnet50(pretrain=pretrain)
        self.base = resnet50_share(pretrain=pretrain)

    def construct(self, x):
        # x = self.base.layer1(x)
        # x = self.base.layer2(x)
        # x = self.base.layer3(x)
        # x = self.base.layer4(x)

        x = self.base(x)

        return x


class embed_net(nn.Cell):
    def __init__(self, low_dim, class_num=200, drop=0.2, part=0, alpha=0.2, nheads=4, arch="resnet50", pretrain=""):
        super(embed_net, self).__init__()
        # print("class_num is :", class_num)
        self.thermal_module = thermal_module(arch=arch, pretrain=pretrain)
        self.visible_module = visible_module(arch=arch, pretrain=pretrain)
        self.base_resnet = base_resnet(arch=arch, pretrain=pretrain)
        pool_dim = 2048
        self.dropout = drop
        self.part = part 

        self.l2norm = Normalize(2)
        self.bottleneck = nn.BatchNorm1d(num_features=pool_dim)
        self.bottleneck.requires_grad=False
        self.classifier = nn.Dense(pool_dim, class_num, has_bias=False)
        # self.classifier1 = nn.Dense(pool_dim, class_num, has_bias=False)
        # self.classifier2 = nn.Dense(pool_dim, class_num, has_bias=False)

        weights_init_kaiming(self.bottleneck)
        weights_init_classifier(self.classifier)

        self.avgpool = P.ReduceMean(keep_dims=True)
        # if self.part > 0:
        #     self.wpa = IWPA(pool_dim, self.part)
        # else:
        #     self.wpa = IWPA(pool_dim, 3)

        self.cat = P.Concat()

    def construct(self, x1, x2=None, adj=None, modal=0, cpa=False):
        
        if modal == 0:
            x1 = self.visible_module(x1)
            x2 = self.thermal_module(x2)
            x = self.cat((x1, x2))
        elif modal == 1:
            x = self.visible_module(x1)
        elif modal == 2:
            x = self.thermal_module(x2)

        # shared four blocks
        # print("x.shape is ", x.shape)
        x = self.base_resnet(x)
        # print("x.shape is ", x.shape)
        x_pool = self.avgpool(x, (2,3))
        # print("x_pool.shape is ", x_pool.shape)
        x_pool = x_pool.view(x_pool.shape[0], x_pool.shape[1])
        # print("After Reshape:", x_pool.shape)
        # print("x_pool is :", x_pool)
        feat = self.bottleneck(x_pool) # mindspore version >=1.3.0
        feat_att = feat

        # if self.part > 0:
        #     # intra_modality weighted part attention
        #     feat_att = self.wpa(x, feat, 1) # why t==1?

        if self.training:
            # cross-modality graph attention
            # TODO: Add cross-modality graph attention mindspore version            
            # return x_pool, self.classifier(feat), self.classifier(feat_att)
            
            out = self.classifier(feat)
            # print("resnet classification output is", out)
            if self.part > 0:
                out_att = self.classifier(feat_att)
                # print("IWPA classification output is", out_att)

            if self.part > 0:
                return feat, feat_att, out, out_att
            else:
                return feat, feat, out, out # just for debug

        else:
            if self.part > 0:
                return self.l2norm(feat), self.l2norm(feat_att)
            else:
                return self.l2norm(feat), self.l2norm(feat) # just for debug


        

