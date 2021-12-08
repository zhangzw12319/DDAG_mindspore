"""attention.py"""
import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.common.initializer as weight_init
import mindspore.ops as P
import mindspore.numpy as msnp
from mindspore.ops import L2Normalize, Transpose
from mindspore.common.initializer import initializer, Constant, XavierUniform


class IWPA(nn.Cell):
    """
    class of IWPA
    """
    def __init__(self, in_channels, part=3, inter_channels=None, out_channels=None):
        super(IWPA, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.out_channels = out_channels
        self.part = part
        self.l2norm = L2Normalize()
        self.softmax = nn.Softmax(axis=-1)
        if ms.__version__ == "1.5.0":
            self.adaptive_pool_2d = P.AdaptiveAvgPool2D((part, 1)) # for mindspore 1.5.0
        else:
            self.adaptive_pool_2d = nn.AvgPool2d((5, 5), (2, 1)) # for mindspore 1.3.0

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

        self.bottleneck = nn.BatchNorm1d(in_channels)
        self.bottleneck.requires_grad = False  # no shift

        # self.bottleneck.weight.set_data(Normal(sigma=0.01))
    # In original PyTorch code:nn.init.normal_(self.bottleneck.weight.data, 1.0, 0.01)

        # weighting vector of the part features
        self.gate = ms.Parameter(
            ms.Tensor(np.ones(self.part), dtype=ms.float32), name="w", requires_grad=True)
        self.gate.set_data(weight_init.initializer(
            Constant(1/self.part), self.gate.shape, self.gate.dtype))

    def construct(self, x, feat, t=None):
        """
        function of constructing
        """
        # x: N x 2048 x 9 x 5
        bt, c = x.shape[:2]
        b = bt // t

        # get part features
        part_feat = self.adaptive_pool_2d(x)
        part_feat = part_feat.view(b, t, c, self.part)
        transpose = Transpose()
        part_feat = transpose(part_feat, (0, 2, 1, 3))  # B, C, T, Part

        part_feat1 = self.fc1(part_feat).view(
            b, self.inter_channels, -1)  # B, C//r, T*part
        part_feat1 = transpose(part_feat1, (0, 2, 1))  # B, T*part, C//r

        part_feat2 = self.fc2(part_feat).view(
            b, self.inter_channels, -1)  # B, C//r, T*part

        part_feat3 = self.fc3(part_feat).view(
            b, self.inter_channels, -1)  # B, C//r, T*part
        part_feat3 = transpose(part_feat3, (0, 2, 1))  # B, T*part, C//r

        # get cross-part attention
        cpa_att = P.matmul(part_feat1, part_feat2)  # B, T*part, T*part
        cpa_att = self.softmax(cpa_att)

        # collect contextual information
        refined_part_feat = P.matmul(cpa_att, part_feat3)  # B, T*Part, C//r
        refined_part_feat = transpose(
            refined_part_feat, (0, 2, 1))  # B, C//r, T*part
        refined_part_feat = refined_part_feat.view(
            (b, self.inter_channels, self.part))  # B, C//r, T, part

        self.gate = self.softmax(self.gate)
        weight_part_feat = P.matmul(refined_part_feat, self.gate)

        weight_part_feat = weight_part_feat + feat
        feat = self.bottleneck(weight_part_feat)

        return feat


class GraphAttentionBlock(nn.Cell):
    """
    Simple GAT layer
    Original idea similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha=0.3, concat=True):
        super(GraphAttentionBlock, self).__init__()
        self.dropout = dropout
        self.drop = nn.Dropout(keep_prob=1-dropout)
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = ms.Parameter(initializer(
            XavierUniform(), (in_features, out_features), ms.float32))
        self.a = ms.Parameter(initializer(
            XavierUniform(), (2 * out_features, 1), ms.float32))

        self.leakyrelu = nn.LeakyReLU(alpha)

    def construct(self, input_features, adj):
        """
        function of constructing
        """
        # input_features: N x in_features
        h = P.matmul(input_features, self.W)  # h: N * out_features
        N = h.shape[0]
        a_input1 = ms.numpy.tile(h, (1, N)).view(
            N * N, -1)  # a_input1: N x (N x out_features)
        a_input2 = ms.numpy.tile(h, (N, 1))  # a_input2: (N x N) x out_features
        a_input = P.Concat(1)([a_input1, a_input2]).view(
            N, -1, 2 * self.out_features)  # a_input: N x N x 2 x out_features
        e = self.leakyrelu(P.matmul(a_input, self.a))
        e = P.Squeeze(2)(e)  # e: N x N

        zero_vec = -9e15 * P.ones_like(e)
        attention = msnp.where(adj > 0, e, zero_vec)
        attention = P.Softmax(1)(attention)
        attention = self.drop(attention)
        h_prime = P.matmul(attention, h)  # h: N x out_features

        if self.concat:
            return P.Elu()(h_prime)
        return h_prime


class GraphAttentionLayer(nn.Cell):
    """
    class of graph_attention layers
    """
    def __init__(self, class_num, nheads, pool_dim, low_dim, dropout=0.2, alpha=0.3):
        super().__init__()
        self.drop = nn.Dropout(keep_prob=1-dropout)
        self.nheads = nheads
        self.attentions = [GraphAttentionBlock(pool_dim, low_dim, dropout=dropout, alpha=alpha, concat=True)
                           for _ in range(nheads)]

        self.out_attention = GraphAttentionBlock(
            low_dim * nheads, class_num, dropout=dropout, alpha=alpha, concat=False)

    def construct(self, input_features, adj):
        feat = self.drop(input_features)
        gall_att = []
        for att in self.attentions:
            gall_att.append(att(feat, adj))
        feat_gatt = P.Concat(1)(gall_att)
        feat_gatt = self.drop(input_features)
        feat_gatt = P.Elu()(self.out_attention(feat_gatt, adj))

        return feat_gatt
