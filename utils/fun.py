import mindspore as ms
import mindspore.nn as nn
import numpy as np

pool = nn.AvgPool2d(kernel_size=3)
x = ms.Tensor(np.random.randint(0, 10,[1, 2, 4, 4]), ms.float32)
output = pool(x)
print(output)