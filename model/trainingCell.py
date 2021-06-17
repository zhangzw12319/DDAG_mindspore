import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as P

from IPython import embed


class MyWithLossCell(nn.Cell):
    def __init__(self, backbone, ce_loss, tri_loss):
        super(MyWithLossCell, self).__init__()
        self._backbone = backbone
        self._ce_loss = ce_loss
        self._tri_loss = tri_loss

        self.cat = P.Concat()
        self.cast = P.Cast()
        self.sum = P.ReduceSum()

    def construct(self, img1, img2, label1, label2, modal=0, cpa=False):
        feat, feat_att, out, out_att = self._backbone(img1, x2=img2, modal=modal, cpa=False)
        label = self.cat((label1,label2))
        label_ = self.cast(label, ms.int32)

        loss_id = self._ce_loss(out, label_)
        # loss_id_att = self._ce_loss(out_att, label_)

        
        loss_id = self.sum(loss_id) / label_.shape[0]

        loss_tri = self._tri_loss(feat, label)
        # loss_tri_att = self._tri_loss(feat_att, label)
        # print("id: {}, tri: {}\r".format(loss_id, loss_tri), end='')
        
        loss_total = loss_id + loss_tri

        return loss_total

    @property
    def backbone_network(self):
        return self._backbone

