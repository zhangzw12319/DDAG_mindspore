"""trainingcell.py"""
import os
import psutil
import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as P
from mindspore import ParameterTuple, Tensor, Parameter
# from mindspore.nn import WithLossCell
# from IPython import embed


def show_memory_info(hint=""):
    pid = os.getpid()

    p = psutil.Process(pid)
    info = p.memory_full_info()
    memory = info.uss/1024./1024
    print(f"{hint} memory used: {memory} MB ")


class CriterionWithNet(nn.Cell):
    """
    class of criterion with network
    """
    def __init__(self, backbone, ce_loss, tri_loss, lossFunc='id'):
        super(CriterionWithNet, self).__init__()
        self._backbone = backbone
        self._ce_loss = ce_loss
        self._tri_loss = tri_loss
        self.lossFunc = lossFunc
        self.acc = Parameter(Tensor(np.array([0]), dtype=ms.float32))
        self.total_loss = Parameter(Tensor(np.array([0]), dtype=ms.float32))
        self.wg = Parameter(Tensor(np.array([0]), dtype=ms.float32))

        # self.total_loss = 0.0
        # self.wg = 0.0

        self.cat = P.Concat()
        self.cast = P.Cast()
        self.sum = P.ReduceSum()
        self.max = P.ArgMaxWithValue(axis=1)
        self.eq = P.Equal()

    def construct(self, img1, img2, label1, label2, adj, modal=0, cpa=False):
        """
        function of constructing
        """
        out_graph = None

        if self._backbone.nheads > 0:
            feat, _, out, out_att, out_graph = self._backbone(
                img1, x2=img2, adj=adj, modal=modal, cpa=False)
        else:
            feat, _, out, out_att = self._backbone(
                img1, x2=img2, modal=modal, cpa=False)

        label = self.cat((label1, label2))
        label_ = self.cast(label, ms.int32)

        loss_id = self._ce_loss(out, label_)
        loss_tri = self._tri_loss(feat, label)

        if self.lossFunc == 'tri':
            loss_total = loss_tri
        elif self.lossFunc == 'id+tri':
            loss_total = loss_id + loss_tri
        else:
            loss_total = loss_id

        if self._backbone.part > 0:
            loss_p = self._ce_loss(out_att, label_)
            loss_total = loss_total + loss_p

        if self._backbone.nheads > 0:
            loss_g = P.NLLLoss("mean")(out_graph, label_,
                                       P.Ones()((out_graph.shape[1]), ms.float32))
            loss_total = loss_total + self.wg * loss_g[0]

        # predict, _ = self.max(out)
        # correct = self.eq(predict, label_)
        # self.acc = ms.numpy.array([ms.numpy.where(correct)[0].shape[0] / label_.shape[0]])
        self.total_loss = loss_total

        return loss_total

    @property
    def backbone_network(self):
        return self._backbone


class OptimizerWithNetAndCriterion(nn.Cell):
    """
    class of optimization methods
    """
    def __init__(self, network, optimizer):
        super(OptimizerWithNetAndCriterion, self).__init__()
        self.network = network
        self.weights = ParameterTuple(optimizer.parameters)
        self.optimizer = optimizer
        self.grad = P.GradOperation(get_by_list=True)

    def set_sens(self, value):
        self.sens = value

    def construct(self, x1, x2, y1, y2, adj):
        weights = self.weights
        grads = self.grad(self.network, weights)(x1, x2, y1, y2, adj)
        # return P.depend(loss, self.optimizer(grads))
        self.optimizer(grads)
        return self.network.total_loss
