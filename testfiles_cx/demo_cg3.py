##################

# @Author: 			   Chenxin Ma
# @Email: 			   machx9@gmail.com
# @Date:               2017-09-23 11:29:17

# @Last Modified by:   Chenxin Ma
# @Last Modified time: 2017-09-24 11:51:23

# @Last modified by:   Heerye
# @Last modified time: 2017-09-24T01:42:50-04:00

##################


from torch.autograd import Variable
import torch
import sys
import os
import numpy as np

sys.path.append(os.path.abspath('..'))
# TODO(xi): can not import data, not found error ... try to resolve the sb issue.
from data.prase_data import libSVM


class ERM():

    def __init__(self, input_dim, num_classes):
        self.model_name = 'ERM'

        self.w1 = Variable(torch.zeros(input_dim, 1), requires_grad=True)
        # self.b1 = Variable(torch.zeros(1, 1), requires_grad=True)
        # self.params = [self.w1, self.b1]
        self.params = [self.w1]

        self.loss = torch.nn.MSELoss(size_average=True)
        self.lmd = 0#1e-3

    def forward(self, x):
    	
        fx = x.mm(self.w1) # + self.b1
        # fx = (torch.t(x) * self.w1)#.add(self.b1)
        return fx

    def get_loss(self, x, y):

        fx = self.forward(x)
        # loss = self.loss(fx, y) * 0.5
        loss = (x.mm(self.w1) - y).pow(2).mean()/2
        # loss = (fx - y).pow(2).sum() / 2/1605
        return loss

    def get_grad(self, x, y):

        out = self.get_loss(x, y)
        g = torch.autograd.grad(out, self.params, create_graph=True)
        for i in range(len(g)):
            g[i].data.add_(self.lmd * self.params[i].data)
        return g

    def get_Hv(self, grad, v):
        gv = 0
        for g_para, v_para in zip(grad, v):
            gv += (g_para * v_para).sum()
        hv = torch.autograd.grad(gv, self.params, create_graph=True)
        for i in range(len(hv)):
            hv[i].data.add_(self.lmd * v[i].data)
        return hv



a = libSVM()
inst = ERM(a.d, 2)
X = Variable(a.X, requires_grad=False)
y = Variable(a.y, requires_grad=False)

# print(-torch.transpose(X,0,1).mm(torch.unsqueeze(y,1))/1605)
grad = inst.get_grad(X, y)
print(grad)
# v = [Variable(para.data) for para in inst.params]
v = []
for para in inst.params:
    v.append(Variable(torch.ones(para.size()[0], 1)))
    # v.append(Variable(torch.from_numpy(np.arange(para.size()[0])/10).type(torch.FloatTensor)))

Ax = inst.get_Hv(grad, v)
# print(Ax)
# verify correctness with Matlab
# print (Ax)
r = []
for para, b in zip(Ax, grad):
    r.append(b - para)

p = [Variable(r_.data) for r_ in r]

r_norm = 0
for r_ in r:
    r_norm += (r_ * r_).sum()
rTr = r_norm

for i in range(10):

    Ap = inst.get_Hv(grad, p)
    pAp = 0
    for p_, Ap_ in zip(p, Ap):
        pAp += (p_ * Ap_).sum()
    alpha = rTr/pAp

    r_norm = 0.0
    for para, p_, r_, Ap_ in zip(inst.params, p, r, Ap):
        para.data.add_(alpha.data * p_.data)
        r_.data.add_(-alpha.data * Ap_.data)
        r_norm += (r_ * r_).sum()

    print(r_norm.data.numpy())
    # stupid if
    if (r_norm < 1e-8).data.numpy():
        print (i)
        break

    rTr_new = r_norm

    beta = rTr_new / rTr
    rTr = rTr_new

    for i in range(len(p)):
        p[i].data = r[i].data + beta.data * p[i].data
