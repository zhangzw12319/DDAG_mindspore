import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as P


class MyWithLossCell(nn.Cell):
    def __init__(self, backbone, ce_loss, tri_loss):
        super(MyWithLossCell, self).__init__()
        self._backbone = backbone
        self._ce_loss = ce_loss
        self._tri_loss = tri_loss

    def construct(self, img1, img2, label1, label2, modal=0, cpa=False):
        feat, feat_att, out, out_att = self._backbone(img1, x2=img2, modal=modal, cpa=False)
        op1 = P.Concat()
        label = op1((label1,label2))
        op2 = P.Cast()
        label_ = op2(label, ms.int32)

        loss_id = self._ce_loss(out, label_)
        # loss_id_att = self._ce_loss(out_att, label_)

        sum = P.ReduceSum()
        loss_id = sum(loss_id) / label_.shape[0]

        # print("loss id is", loss_id)

        loss_tri = self._tri_loss(feat, label)
        # loss_tri_att = self._tri_loss(feat_att, label)
        # print("triplet id is", loss_tri)
        
        return loss_id + loss_tri

    @property
    def backbone_network(self):
        return self._backbone

