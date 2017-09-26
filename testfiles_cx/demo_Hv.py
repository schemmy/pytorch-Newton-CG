##################

# @Author: 			   Chenxin Ma
# @Email: 			   machx9@gmail.com
# @Date:               2017-09-21 20:40:16
# @Last modified by:   Heerye
# @Last modified time: 2017-09-24T14:14:11-04:00

##################


import torch
import numpy as np
from torch.autograd import Variable, grad

A = np.array([[1, 2], [3, 4], [5, 6]])
A = Variable(torch.from_numpy(A).type(torch.FloatTensor))
x = Variable(torch.ones(2, 1), requires_grad=True)
y = Variable(torch.ones(1, 1) * 2, requires_grad=True)

out = (A.mm(x).add(y)).pow(2).mean()

print(out)

g = grad(out, [x, y], create_graph=True)
print(g)

v = []
for para in [x, y]:
    v.append(Variable(torch.ones(para.size())))

gv = 0.0
for i, j in zip(g, v):
    gv += (i * j).sum()

hv = grad(gv, [x, y])
print(hv)
