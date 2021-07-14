import mindspore.nn as nn
import mindspore.ops as P
import mindspore.common.initializer as weight_init
from model.resnet import *
from model.attention import IWPA

import os
import psutil
from IPython import embed

def show_memory_info(hint=""):
    pid = os.getpid()

    p = psutil.Process(pid)
    info = p.memory_full_info()
    memory = info.uss/1024./1024
    print(f"{hint} memory used: {memory} MB ")

class Normalize(nn.Cell):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def construct(self, x):
        pow = P.Pow()
        sum = P.ReduceSum()
        div = P.Div()

        norm = pow(x, self.power)
        norm = sum(x)
        norm = pow(norm, 1. / self.power)
        out = div(x, norm)
        return out


class visible_module(nn.Cell):
    def __init__(self, arch="resnet50", ifPretrained=False, pretrainedPath=None):
        super(visible_module, self).__init__()

        # self.visible = resnet50()
        if ifPretrained and pretrainedPath is not None:
            self.visible = resnet50_specific(ifPretrained=True, pretrainedPath=pretrainedPath)
        else:
            self.visible = resnet50_specific()
    def construct(self, x):
        # x = self.visible.conv1(x)
        # x = self.visible.bn1(x)
        # x = self.visible.relu(x)
        # x = self.visible.maxpool(x)

        x = self.visible(x)

        return x


class thermal_module(nn.Cell):
    def __init__(self, arch="resnet50", ifPretrained=False, pretrainedPath=None):
        super(thermal_module, self).__init__()

        # self.thermal = resnet50()
        if ifPretrained and pretrainedPath is not None:
            self.thermal = resnet50_specific(ifPretrained=True, pretrainedPath=pretrainedPath)
        else:
            self.thermal = resnet50_specific()

    def construct(self, x):
        # x = self.thermal.conv1(x)
        # x = self.thermal.bn1(x)
        # x = self.thermal.relu(x)
        # x = self.thermal.maxpool(x)

        x = self.thermal(x)

        return x


class base_resnet(nn.Cell):
    def __init__(self, arch="resnet50", ifPretrained=False, pretrainedPath=None):
        super(base_resnet, self).__init__()

        # self.base = resnet50()
        if ifPretrained and pretrainedPath is not None:
            self.base = resnet50_share(ifPretrained=True, pretrainedPath=pretrainedPath)
        else:
            self.base = resnet50_share()

    def construct(self, x):
        # x = self.base.layer1(x)
        # x = self.base.layer2(x)
        # x = self.base.layer3(x)
        # x = self.base.layer4(x)

        x = self.base(x)

        return x


class embed_net(nn.Cell):
    def __init__(self, low_dim, class_num=200, drop=0.2, part=0, alpha=0.2, 
        nheads=4, arch="resnet50", ifPretrained=False, pretrainedPath=None):
        super(embed_net, self).__init__()
        # print("class_num is :", class_num)
        if ifPretrained:
            self.thermal_module = thermal_module(arch=arch, ifPretrained=ifPretrained, pretrainedPath=pretrainedPath)
            self.visible_module = visible_module(arch=arch, ifPretrained=ifPretrained, pretrainedPath=pretrainedPath)
            self.base_resnet = base_resnet(arch=arch, ifPretrained=ifPretrained, pretrainedPath=pretrainedPath)
        else:
            self.thermal_module = thermal_module(arch=arch)
            self.visible_module = visible_module(arch=arch)
            self.base_resnet = base_resnet(arch=arch)
        pool_dim = 2048
        self.dropout = drop
        self.part = part 

        self.l2norm = Normalize(2)
        self.bottleneck = nn.BatchNorm1d(num_features=pool_dim)
        self.bottleneck.requires_grad=False # Maybe problematic? Original in PyTorch bottleneck.bias.requires_grad(False)

        self.classifier = nn.Dense(pool_dim, class_num, has_bias=False)
        # self.classifier1 = nn.Dense(pool_dim, class_num, has_bias=False)
        # self.classifier2 = nn.Dense(pool_dim, class_num, has_bias=False)

        # TODO:add weights initialization module
        # self.bottleneck.apply(weights_init_kaiming)
        # self.classifier.apply(weights_init_classifier)
        # self.classifier1.apply(weights_init_classifier)
        # self.classifier2.apply(weights_init_classifier)

        
        self.classifier.weight.set_data(
           weight_init.initializer(weight_init.Normal(sigma=0.001), \
           self.classifier.weight.shape, self.classifier.weight.dtype))
        
        self.avgpool = P.ReduceMean(keep_dims=True)
        if self.part > 0:
            self.wpa = IWPA(pool_dim, self.part)
        else:
            self.wpa = IWPA(pool_dim, 3)

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
        # feat = self.bottleneck(x_pool) # do not support cpu
        feat = x_pool

        if self.part > 0:
            # intra_modality weighted part attention
            feat_att = self.wpa(x, feat, 1) # why t==1?

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


        

