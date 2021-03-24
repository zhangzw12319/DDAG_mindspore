import mindspore as ms
import mindspore.nn as nn


class MyWithLossCell(nn.Cell):
    def __init__(self, backbone, loss_fn):
        super(MyWithLossCell, self).__init__()
        self._backbone = backbone
        self._loss_fn = loss_fn

    def construct(self, img1, img2, label1, label2, modal=0, cpa=False):
        feat, feat_att, out, out_att = self._backbone(img1, x2=img2, modal=modal, cpa=False)
        loss_id = self._loss_fn(out, label1)
        loss_id_att = self._loss_fn(out_att, label2)
        return loss_id + loss_id_att

    @property
    def backbone_network(self):
        return self._backbone

