import mindspore as ms
import mindspore.nn as nn
import numpy as np
import mindspore.common.initializer as weight_init
import mindspore.ops as P

from mindspore import Tensor
from mindspore.common.initializer import Normal, Constant
 


# net = nn.MatMul()
# input_x1 = Tensor(np.ones(shape=[3, 2, 3]), ms.float32)
# input_x2 = Tensor(np.ones(shape=[2, 3, 4]), ms.float32)
# output = net(input_x1, input_x2)
# print(output.shape)

# ------------------------------------------------------------

gate = ms.Parameter(ms.Tensor(np.ones(3), dtype=ms.float64), name="w", requires_grad=True)
gate.set_data(weight_init.initializer(Constant(1/3), gate.shape, gate.dtype))
print(gate.dtype)
print("gate is ", gate)
softmax = P.Softmax()
gate_ = softmax(gate)
print(gate_)