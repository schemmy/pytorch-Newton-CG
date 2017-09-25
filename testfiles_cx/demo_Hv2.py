##################

# @Author: 			   Xi He
# @Email:
# @Date:               2017-09-23 01:29:28
# @Last modified by:   Heerye
# @Last modified time: 2017-09-24T01:40:06-04:00

##################

import torch
from torch.autograd import Variable, grad
import torch.nn.functional as F
import numpy as np
torch.manual_seed(1)

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y = x.pow(2) + 0.2 * torch.rand(x.size())
x, y = Variable(x), Variable(y)

HIDDEN = 4
thre = np.sqrt(6.0 / (1 + HIDDEN))
w1 = Variable(torch.ones(1, HIDDEN) * thre, requires_grad=True)
b1 = Variable(torch.ones(x.size()[0], HIDDEN) * 0.1, requires_grad=True)
w2 = Variable(torch.rand(HIDDEN, 1) * thre, requires_grad=True)
b2 = Variable(torch.ones(x.size()[0], 1) * 0.1, requires_grad=True)

params = [w1, b1, w2, b2]

# TODO(xi&chenxin): Hv is all 1 for relu activation. How about sigmoid, should Hv equal to all 1 as well?
# TODO(xi&chenxin): Can we use relu in second-order methods?
# h = F.sigmoid(torch.mm(x, w1) + b1)
h = F.relu(torch.mm(x, w1) + b1)

out = torch.mm(h, w2) + b2

loss = torch.mean((out - y).pow(2))
print(loss)

g = grad(loss, params, create_graph=True)
print(g)

v = []
for para in params:
    v.append(Variable(torch.ones(para.size()[0], 1)))

gv = 0
for g_para, v_para in zip(params, v):
    gv += (g_para * v_para).sum()

hv = grad(gv, params)
print(hv)
