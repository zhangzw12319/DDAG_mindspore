import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.common.initializer as weight_init
import mindspore.ops as P
from mindspore.ops import L2Normalize, Transpose
from mindspore.common.initializer import Normal, Constant


class IWPA(nn.Cell):
    def __init__(self, in_channels, part=3, inter_channels=None, out_channels=None):
        super(IWPA, self).__init__()
        
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.out_channels = out_channels
        self.part = part
        self.l2norm = L2Normalize()
        self.softmax = nn.Softmax(axis=-1)

        if self.inter_channels is None:
            self.inter_channels = in_channels

        if self.out_channels is None:
            self.out_channels = in_channels
        
        self.fc1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
        kernel_size=1, stride=1, padding=0)

        self.fc2 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
        kernel_size=1, stride=1, padding=0)

        self.fc3 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
        kernel_size=1, stride=1, padding=0)

        self.W = nn.SequentialCell(
            nn.Conv2d(in_channels=self.inter_channels, out_channels=self.out_channels,
            kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.out_channels),
        )
        # self.W[1].weight.set_data(Constant(0.0))
        # self.w[2].bias.set_data(Constant(0.0))

        self.bottleneck = nn.BatchNorm1d(in_channels)
        self.bottleneck.requires_grad=False # no shift

        # self.bottleneck.weight.set_data(Normal(sigma=0.01)) 
    #In original PyTorch code:nn.init.normal_(self.bottleneck.weight.data, 1.0, 0.01)
       

        # weighting vector of the part features
        self.gate = ms.Parameter(ms.Tensor(np.ones(self.part), dtype=ms.float32), name="w", requires_grad=True)
        self.gate.set_data(weight_init.initializer(Constant(1/self.part), self.gate.shape, self.gate.dtype))


    def construct(self, x, feat, t=None): # ? what is t?
        bt, c, h ,w = x.shape
        b = bt // t

        # get part features
        part_feat_pool = nn.AvgPool2d(kernel_size=(6,9), stride=(6,1))
        part_feat = part_feat_pool(x)
        part_feat = part_feat.view(b, t, c, self.part)
        transpose =  Transpose()
        part_feat = transpose(part_feat, (0, 2, 1, 3)) # B, C, T, Part

        part_feat1 = self.fc1(part_feat).view(b, self.inter_channels, -1) # B, C//r, T*part
        part_feat1 = transpose(part_feat1, (0, 2, 1)) # B, T*part, C//r

        part_feat2 = self.fc2(part_feat).view(b, self.inter_channels, -1) # B, C//r, T*part

        part_feat3 = self.fc3(part_feat).view(b, self.inter_channels, -1) # B, C//r, T*part
        part_feat3 = transpose(part_feat3, (0, 2, 1)) # B, T*part, C//r

        # get cross-part attention
        cpa_att = mat_mul(part_feat1, part_feat2) # B, T*part, T*part
        cpa_att = self.softmax(cpa_att)

        # collect contextual information
        refined_part_feat = P.matmul(cpa_att, part_feat3) # B, T*Part, C//r
        refined_part_feat = transpose(refined_part_feat, (0, 2, 1)) # B, C//r, T*part
        refined_part_feat = refined_part_feat.view((b, self.inter_channels, self.part)) # B, C//r, T, part

        # gate = self.softmax(self.gate)
        # weight_part_feat = nn.MatMul(refined_part_feat, gate)
        weight_part_feat = P.matmul(refined_part_feat, self.gate)
        weight_part_feat = weight_part_feat.view((weight_part_feat.shape[0], weight_part_feat.shape[1], 1 ,1))
        print("weight_part_feat shape is", weight_part_feat.shape)
        
        weight_part_feat = weight_part_feat + feat
        feat = self.bottleneck(weight_part_feat)

        return feat