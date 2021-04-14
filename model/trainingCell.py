import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as P


class MyWithLossCell(nn.Cell):
    def __init__(self, backbone, loss_fn):
        super(MyWithLossCell, self).__init__()
        self._backbone = backbone
        self._loss_fn = loss_fn

    def construct(self, img1, img2, label1, label2, modal=0, cpa=False):
        feat, feat_att, out, out_att = self._backbone(img1, x2=img2, modal=modal, cpa=False)
        op1 = P.Concat()
        label = op1((label1,label2))
        op2 = P.Cast()
        label_ = op2(label, ms.int32)
        print("label", label_.dtype)
        loss_id = self._loss_fn(out, label_)
        loss_id_att = self._loss_fn(out_att, label_)
        
        return loss_id + loss_id_att

    @property
    def backbone_network(self):
        return self._backbone

