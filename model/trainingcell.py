import os
import psutil
import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as P
from mindspore import ParameterTuple, Tensor
from mindspore.nn import WithLossCell


from IPython import embed

def show_memory_info(hint=""):
    pid = os.getpid()

    p = psutil.Process(pid)
    info = p.memory_full_info()
    memory = info.uss/1024./1024
    print(f"{hint} memory used: {memory} MB ")

class CriterionWithNet(nn.Cell):
    def __init__(self, backbone, ce_loss, tri_loss, lossFunc='id'):
        super(CriterionWithNet, self).__init__()
        self._backbone = backbone
        self._ce_loss = ce_loss
        self._tri_loss = tri_loss
        self.lossFunc = lossFunc
        self.acc = 0
        self.total_loss = None
        self.wg = 0

        self.cat = P.Concat()
        self.cast = P.Cast()
        self.sum = P.ReduceSum()
        self.max = P.ArgMaxWithValue(axis=1)
        self.eq = P.Equal()

    def construct(self, img1, img2, label1, label2, modal=0, adj=None, cpa=False):

        if self._backbone.nheads > 0:
            feat, _ , out, out_att, out_graph = self._backbone(img1, x2=img2, adj=adj, modal=modal, cpa=False)
        else:
            feat, _ , out, out_att = self._backbone(img1, x2=img2, modal=modal, cpa=False)

        label = self.cat((label1, label2))
        label_ = self.cast(label, ms.int32)

        np.save("batchdata.npy", feat.asnumpy())
        np.save("batchlabel.npy", label_.asnumpy())

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
            loss_total += loss_p

        if self._backbone.nheads > 0:
            loss_g = P.NLLLoss("mean")(out_graph, label_, P.Ones()((out_graph.shape[1]), ms.float32))
            loss_total += self.wg * loss_g[0]
  
        predict , _ = self.max(out)
        correct = self.eq(predict, label_)
        self.acc = np.where(correct)[0].shape[0] / label_.shape[0]
        self.total_loss = loss_total

        return loss_total

    @property
    def backbone_network(self):
        return self._backbone

class OptimizerWithNetAndCriterion(nn.Cell):
    def __init__(self, network, optimizer):
        super(OptimizerWithNetAndCriterion, self).__init__()
        self.network = network
        self.weights = ParameterTuple(optimizer.parameters)
        self.optimizer = optimizer
        self.grad = P.GradOperation(get_by_list=True)

    def set_sens(self, value):
        self.sens = value

    def construct(self, *inputs, **kwargs):
        weights = self.weights
        grads = self.grad(self.network, weights)(*inputs, **kwargs)
        # return P.depend(loss, self.optimizer(grads))
        self.optimizer(grads)
        return self.network.total_loss
