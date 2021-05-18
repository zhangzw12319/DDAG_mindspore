import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as P
import numpy as np
from mindspore import Tensor


class OriTripletLoss(nn.Cell):
    """Triplet loss with hard positive/negative mining.

    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Args:
    - margin (float): margin for triplet.
    """

    def __init__(self, margin=0.3):
        super(OriTripletLoss, self).__init__()
        self.margin = margin

    def ranking_loss(self, input1, input2, y):
        sub = P.Sub()
        mul = P.Mul()
        add = P.Add()

        temp1 = -sub(input1, input2)
        temp2 = mul(temp1, y)
        temp3 = add(temp2, self.margin)
        temp3_mask = np.greater_equal(temp3, 0)

        loss = 0
        for i in range(temp3.shape[0]):
            if temp3_mask[i]:
                loss += temp3[i]


        loss = Tensor(loss / temp3.shape[0])
        # print(loss)
        return loss

    def construct(self, inputs, targets):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """
        n = inputs.shape[0]

        # Compute pairwise distance, replace by the official when merged
        pow = P.Pow()
        sum = P.ReduceSum(keep_dims=True)
        expand = P.BroadcastTo((n, n))
        transpose = P.Transpose()
        mul = P.Mul()
        add = P.Add()
        sqrt = P.Sqrt()
        equal = P.Equal()
        cat = P.Concat()
        ones_like = P.OnesLike()

        dist = pow(inputs, 2)
        dist = sum(dist, axis=1)
        dist = expand(dist)
        dist = dist + transpose(dist, (1, 0))

        temp1 = P.matmul(inputs, transpose(inputs, (1, 0)))
        temp1 = mul(-2, temp1)
        dist = add(dist, temp1)
        dist = P.composite.clip_by_value(dist, clip_value_min=1e-12, clip_value_max=100000000)  # for numerical stability, clip_value_max=? why must set?
        dist = sqrt(dist)

        # For each anchor, find the hardest positive and negative
        targets = expand(targets)
        mask = equal(targets, transpose(targets, (1, 0)))
        dist_ap = []
        dist_an = []

        # only for debugging
        #####################
        # print("dist is")
        # print(dist.shape)
        # print(dist)
        # print("mask is")
        # print(mask.shape)
        # print(mask)
        # print(mask[0])
        #####################

        for i in range(n):
            minval = -1.0
            maxval = -1.0
            for j in range(n):
                if mask[i][j] and dist[i][j] > maxval:
                    maxval = dist[i][j]
            dist_ap.append(maxval.asnumpy())

            for j in range(n):
                if not mask[i][j] and (dist[i][j] < minval or minval == -1):
                    minval = dist[i][j]
            dist_an.append(minval.asnumpy())

        dist_ap = Tensor(dist_ap, ms.float32)
        dist_an = Tensor(dist_an, ms.float32)
        # only for debugging
        #####################
        # print(dist_ap)
        # print(dist_ap.shape)
        # print(dist_an)
        #####################

        # Compute ranking hinge loss
        y = ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)

        # # compute accuracy
        # correct = torch.ge(dist_an, dist_ap).sum().item()
        return loss


# class GradOriTripletLoss(nn.Cell)
#     def __init__(self, net):
#         super(GradOriTripletLoss, self).__init__()
#         self.net = net
#         self.grad_op = P.GradOperation(get_all=True)
#
#     def construct(self, inputs, targets):
#         gradient_function = self.grad_op(self.net)
#         return gradient_function(inputs, targets)