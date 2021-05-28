import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as P
import numpy as np
from mindspore import Tensor

from IPython import embed

class MarginRankingLoss(nn.Cell):
    def __init__(self, margin=0, error_msg=None):
        super(MarginRankingLoss, self).__init__()
        self.margin = margin
        self.error_msg = error_msg

    def construct(self, input1, input2, y):
        sub = P.Sub()
        mul = P.Mul()
        add = P.Add()
        ge = P.GreaterEqual()

        temp1 = -sub(input1, input2)
        temp2 = mul(temp1, y)
        temp3 = add(temp2, self.margin)
        temp3_mask = ge(temp3, 0)

        loss = Tensor()
        for i in range(temp3.shape[0]):
            if temp3_mask[i]:
                loss += temp3[i]

        loss = Tensor(loss / temp3.shape[0])
        return loss

class OriTripletLoss(nn.Cell):
    """Triplet loss with hard positive/negative mining.

    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Args:
    - margin (float): margin for triplet.
    """

    def __init__(self, margin=0.3, error_msg=None):
        super(OriTripletLoss, self).__init__()
        self.margin = margin
        self.error_msg = error_msg

    def ranking_loss(self, input1, input2, y):
        sub = P.Sub()
        mul = P.Mul()
        add = P.Add()
        ge = P.GreaterEqual()

        temp1 = -sub(input1, input2)
        temp2 = mul(temp1, y)
        temp3 = add(temp2, self.margin)
        # temp3_mask = np.greater_equal(temp3, 0)
        temp3_mask = ge(temp3, 0)


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
                if not mask[i][j] and (dist[i][j] < minval or minval == -1):
                    minval = dist[i][j]

            if(not isinstance(minval, Tensor) or not isinstance(maxval, Tensor)
                    or minval == -1.0 or maxval == -1.0):
                if self.error_msg is not None:
                    print("Error Msg", file=self.error_msg)
                    print("mask {} is".format(i), file=self.error_msg)
                    print(mask[i], file=self.error_msg)
                    print("dist is:", file=self.error_msg)
                    print(dist[i], file=self.error_msg)
                    print(maxval, file=self.error_msg)
                    print(minval, file=self.error_msg)
                    print(type(maxval), file=self.error_msg)
                    print(type(minval), file=self.error_msg)
                    self.error_msg.flush()

            # assert minval != -1.0 and isinstance(minval, Tensor)
            # assert maxval != -1.0 and isinstance(maxval, Tensor)
            dist_ap.append(maxval.asnumpy())
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

# class TripletLoss(nn.Cell):
#     """Triplet loss with hard positive/negative mining.
    
#     Reference:
#     Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
#     Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
    
#     Args:
#     - margin (float): margin for triplet.
#     """
#     def __init__(self, margin=0.3, error_msg=None):
#         super(TripletLoss, self).__init__()
#         self.margin = margin
#         self.error_msg = error_msg
#         self.ranking_loss = MarginRankingLoss(margin=margin) #######################
#         self.cat = P.Concat()
#         self.eye = P.Eye()
#         self.ones_like = P.OnesLike()
#         self.ge = P.GreaterEqual()
#         self.unsqueeze = P.ExpandDims()
#         self.max =  P.Maximum()
#         self.min =  P.Minimum()

#     def pdist_torch(emb1, emb2):
#         '''
#         compute the eucilidean distance matrix between embeddings1 and embeddings2
#         using gpu
#         '''
#         pow = P.Pow()
#         sum = P.ReduceSum(dim = 1, keep_dims=True)
#         m, n = emb1.shape[0], emb2.shape[0]
#         expand = P.BroadcastTo()
#         # emb1_pow = expand(sum(pow(emb1, 2)), m, n)  #######################
#         # emb2_pow = expand(sum(pow(emb2, 2)), n, m).t()  #######################
#         # dist_mtx = emb1_pow + emb2_pow  #######################
#         # dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())  #######################
#         # # dist_mtx = dist_mtx.clamp(min = 1e-12)  #######################
#         # dist_mtx = dist_mtx.clamp(min = 1e-12).sqrt()  #######################
#         return dist_mtx    
    
#     def construct(self, input, target):
#         """
#         Args:
#         - input: feature matrix with shape (batch_size, feat_dim)
#         - target: ground truth labels with shape (num_classes)
#         """
        
#         n = input.shape[0]
#         input1 = input[:n, :]
#         input2 = input[n:n+n, :]
#         # input1 = input.narrow(0,0,n)  #######################
#         # input2 = input.narrow(0,n,n)  #######################
#         mask = self.eye(n, n, ms.int32)
        
#         # Compute pairwise distance, replace by the official when merged
#         dist = self.pdist_torch(input1, input2)
        
#         # For each anchor, find the hardest positive and negative
#         # mask = target1.expand(n, n).eq(target1.expand(n, n).t())
#         dist_ap, dist_an = [], []
#         for i in range(n):
#             dist_ap.append(self.unsqueeze(dist[i,i], 0))
#             dist_an.append(self.unsqueeze(self.min(dist[i][mask[i] == 0]), 0))
#         dist_ap = self.cat(dist_ap)
#         dist_an = self.cat(dist_an)
        
#         # Compute ranking hinge loss
#         y = self.ones_like(dist_an)
#         loss = self.ranking_loss(dist_an, dist_ap, y)
        
#         # compute accuracy
#         correct = self.sum(self.ge(dist_an, dist_ap)) * 2
#         return loss
