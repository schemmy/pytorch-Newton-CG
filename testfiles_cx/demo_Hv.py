##################

# @Author: 			   Chenxin Ma
# @Email: 			   machx9@gmail.com
# @Date:               2017-09-21 20:40:16
# @Last modified by:   Heerye
# @Last modified time: 2017-09-22T14:20:40-04:00

##################


import torch
import numpy as np
from torch.autograd import Variable, grad

x = Variable(torch.ones(2), requires_grad=True)
y = Variable(torch.ones(2), requires_grad=True)
out = (x * x + y * y).mean()
g = grad(out, [x, y], create_graph=True)

# out.backward(retain_graph = True)
# v = Variable(torch.from_numpy(np.array([2, 3]))).type(torch.FloatTensor)
v = []
for para in [x, y]:
    v.append(Variable(torch.ones(para.size())))

gv = 0.0
for i, j in zip(g, v):
    gv += (i * j).sum()

hv = grad(gv, [x, y])
print([i.data for i in hv])
# print(v, g)
# gv = (g[0] * v).sum()
# hv = grad(gv, x)
# print(hv[0].data)
