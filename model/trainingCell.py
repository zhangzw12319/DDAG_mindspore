import os
import psutil
import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as P
from mindspore import ParameterTuple
from mindspore.nn import WithLossCell


from IPython import embed

def show_memory_info(hint=""):
    pid = os.getpid()

    p = psutil.Process(pid)
    info = p.memory_full_info()
    memory = info.uss/1024./1024
    print(f"{hint} memory used: {memory} MB ")

class Criterion_with_Net(nn.Cell):
    def __init__(self, backbone, ce_loss, tri_loss):
        super(Criterion_with_Net, self).__init__()
        self._backbone = backbone
        self._ce_loss = ce_loss
        self._tri_loss = tri_loss

        self.cat = P.Concat()
        self.cast = P.Cast()
        self.sum = P.ReduceSum()

    def construct(self, img1, img2, label1, label2, modal=0, cpa=False):

        feat, feat_att, out, out_att = self._backbone(img1, x2=img2, modal=modal, cpa=False)
        label = self.cat((label1, label2))
        label_ = self.cast(label, ms.int32)

        loss_id = self._ce_loss(out, label_)
        # loss_id_att = self._ce_loss(out_att, label_)

        loss_tri = self._tri_loss(feat, label)
        # loss_tri_att = self._tri_loss(feat_att, label)
        # print("id: {}, tri: {}\r".format(loss_id, loss_tri), end='')
        # print("id: {}".format(loss_id))
        loss_total = loss_id + loss_tri

        return loss_total

    @property
    def backbone_network(self):
        return self._backbone

class Optimizer_with_Net_and_Criterion(nn.Cell):
    def __init__(self, network, optimizer, sens=1.0):
        super(Optimizer_with_Net_and_Criterion, self).__init__(auto_prefix=True)
        self.network = network
        self.weights = ParameterTuple(network.trainable_params())
        self.optimizer = optimizer
        self.grad = P.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens

    def set_sens(self, value):
        self.sens = value

    def construct(self, *inputs):
        weights = self.weights
        loss = self.network(*inputs)
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(*inputs, sens)
        # print(np.sum(self.network.trainable_params()[0].asnumpy()))
        # for i in range(len(self.network.trainable_params())):
        #     print(np.sum(self.network.trainable_params()[i].asnumpy()))
        return P.depend(loss, self.optimizer(grads))
