import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.common.initializer as weight_init
from resnet import *
from attention import IWPA


class Normalize(nn.Cell):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def construct(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class visible_module(nn.Cell):
    def __init__(self, arch="resnet50"):
        super(visible_module, self).__init__()

        model_v = resnet50(last_conv_stride=1, last_conv_dilation=1)

        self.visible = model_v

    def construct(self, x):
        x = self.visible.conv1(x)
        x = self.visible.bn1(x)
        x = self.visible.relu(x)
        x = self.visible.maxpool(x)

        return x


class thermal_module(nn.Cell):
    def __init__(self, arch="resnet50"):
        super(thermal_module, self).__init__()
        
        model_t = resnet50(last_conv_stride=1, last_conv_dilation=1)
        self.thermal = model_t

    def construct(self, x):
        x = self.thermal.conv1(x)
        x = self.thermal.bn1(x)
        x = self.thermal.relu(x)
        x = self.thermal.maxpool(x)

        return x


class base_resnet(nn.Cell):
    def __init__(self, arch="resnet50")():
        super(base_resnet, self).__init__()
        model_base = resnet50(last_conv_dilation=1, last_conv_dilation=1)
        # avg pooling to global pooling
        model_base.avgpool = nn.AvgPool2d((1,1))
        self.base = model_base

    def construct(self, x):
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)

        return x




class embed_net(nn.Cell):
    def __init__(self, low_dim, class_num, drop=0.2, part=3, alpha=0.2, nheads=4, arch="resnet50", wpa=False):
        super(embed_net, self).__init__()

        self.thermal_module=thermal_module(arch=arch)
        self.visible_module=visible_module(arch=arch)
        self.base_resnet = base_resnet(arch=arch)
        pool_dim = 2048
        self.dropout = drop
        self.part = part
        self.lpa = wpa

        self.l2norm = Normalize(2)
        self.bottleneck = nn.BatchNorm1d(pool_dim)
        self.bottleneck.requires_grad(False) # Maybe problematic? Original in PyTorch bottleneck.bias.requires_grad(False)

        self.classifier = nn.Dense(pool_dim, class_num, has_bias=False)
        # self.classifier1 = nn.Dense(pool_dim, class_num, has_bias=False)
        # self.classifier2 = nn.Dense(pool_dim, class_num, has_bias=False)

        # TODO:add weights initialization module
        # self.bottleneck.apply(weights_init_kaiming)
        # self.classifier.apply(weights_init_classifier)
        # self.classifier1.apply(weights_init_classifier)
        # self.classifier2.apply(weights_init_classifier)

        self.bottleneck.weight.set_data(weight_init.Normal(sigma=0.01), self.bottleneck.weight.shape, self.bottleneck.weight.dtype)
        self.classifier.weight.set_data(weight_init.Normal(sigma=0.001), self.classifier.weight.shape, self.classifier.weight.dtype)
        self.classifier.bias.set_data(weight_init.Zero(), self.classifier.bias.shape, self.classifier.bias.dtype)
        
        self.avgpool = nn.AvgPool2d((1,1))
        self.wpa = IWPA(pool_dim, part)

    def forward(self, x1,x2, adj=None, modal=0, cpa = False):
        # domain specific block
        if modal == 0:
            x1 = self.visible_module(x1)
            x2 = self.thermal_module(x2)
            cat_op = ops.Concat()
            x = cat_op((x1, x2), 0)
        elif modal == 1:
            x = self.visible_module(x1)
        elif modal == 2:
            x = self.thermal_module(x2)

        # shared four blocks
        x = self.base_resnet(x)
        x_pol = self.avgpool(x)
        x_pool = x_pool.view(x_pool.size(0), x_pool.size(1))
        feat = self.bottleneck(x_pool)

        if self.lpa:
            # intra_modality weighted part attention
            feat_att = self.wpa(x, feat, 1, self.part) # why t==1?

        if self.training:
            # cross-modality graph attention
            # TODO: Add cross-modality graph attention mindspore version
            # pass
            
            return x_pool, self.classifier(feat), self.classifier(feat_att)

        else:
            return self.l2norm(feat), self.l2norm(feat_att)

        

