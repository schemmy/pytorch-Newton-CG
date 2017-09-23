##################

# @Author: 			   Chenxin Ma
# @Email: 			   machx9@gmail.com
# @Date:               2017-09-21 20:43:33
# @Last modified by:   Heerye
# @Last modified time: 2017-09-22T10:45:49-04:00

##################


import sys
import os
sys.path.append(os.path.abspath('../data'))
from dataset import Mnist
from prase_data import libSVM
sys.path.append(os.path.abspath('../models'))
from ERM import ERM

import torch
import numpy as np
from torch.autograd import Variable




a = libSVM()
inst = ERM(a.d, 2)
X = Variable(a.X, requires_grad=False)
y = Variable(a.y, requires_grad=False)
grad = inst.get_grad(X, y)

v = [Variable(para.data) for para in inst.params]
Ax = inst.get_Hv(grad, v)

r = []
for para, b in zip(Ax, grad):
    r.append(b - para)


p = [Variable(r_.data) for r_ in r]

r_norm = 0
for r_ in zip(r):
    r_norm += r_[0].norm()
rTr = r_norm ** 2

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
        r_norm += r_.norm()

    # stupid if
    if (r_norm < 1e-4).data.numpy():
        break

    print(r_norm)
    rTr_new = r_norm ** 2

    beta = rTr_new / rTr
    rTr = rTr_new

    for i in range(len(p)):
        p[i].data = r[i].data + beta.data * p[i].data




'''

for i in range(1):
    b.zero_grad()
    output =  b.get_loss(X, y)
    # print b.linear.weight.grad
    # for param in b.parameters():
        # param.data.add_(-0.1 * param.grad.data)

    pa = [b.linear.weight, b.linear.bias]
    g = torch.autograd.grad(output, pa, create_graph=True)

    # output.backward(retain_graph = True)
    # g = b.linear.weight.grad, b.linear.bias.grad]
    # print g[0].data

    v = []
    for para in b.parameters():
        v.append(Variable(torch.ones(para.size())))


    gv = 0
    for g_para, v_para in zip(g, v):
        gv += (g_para * v_para).sum()

    hv = torch.autograd.grad(gv, pa)
    print (hv)
# print output
'''
