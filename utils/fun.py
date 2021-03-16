import mindspore as ms
import mindspore.nn as nn
import numpy as np
from mindspore import Tensor


net = nn.MatMul()
input_x1 = Tensor(np.ones(shape=[3, 2, 3]), ms.float32)
input_x2 = Tensor(np.ones(shape=[2, 3, 4]), ms.float32)
output = net(input_x1, input_x2)
print(output.shape)

