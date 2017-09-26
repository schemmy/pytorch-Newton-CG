##################

# @Author: 			   Chenxin Ma
# @Email: 			   machx9@gmail.com
# @Date:               2017-09-21 20:43:33
# @Last modified by:   Heerye
# @Last modified time: 2017-09-25T14:02:44-04:00

##################


import sys
import os
sys.path.append(os.path.abspath('..'))
from data.dataset import Mnist
from data.prase_data import libSVM
from models.ERM import ERM

import torch
import numpy as np
from torch.autograd import Variable
import time

def CG(inst, grad):

    inst.params = inst.get_params()

    x = [Variable(torch.zeros(para.size()) ) for para in inst.params]
    Ax = inst.get_Hv(grad, x)

    r = []
    for para, b in zip(Ax, grad):
        r.append(b - para)


    p = [Variable(r_.data.clone()) for r_ in r]

    r_norm = 0
    for r_ in r:
        r_norm += (r_ * r_).sum()
    rTr = r_norm

    iter_count = 0
    while(1):
        iter_count += 1

        Ap = inst.get_Hv(grad, p)

        pAp = 0
        for p_, Ap_ in zip(p, Ap):
            pAp += (p_ * Ap_).sum()
        alpha = rTr/pAp

        r_norm = 0.0
        for x_, p_, r_, Ap_ in zip(x, p, r, Ap):
            x_.data.add_(alpha.data * p_.data)
            r_.data.add_(-alpha.data * Ap_.data)
            r_norm += (r_ * r_).sum()

        # print(r_norm.data.numpy())
        # stupid if
        if (r_norm < 1e-10).data.numpy() or iter_count >= 10:
            # print (i)
            break

        rTr_new = r_norm

        beta = rTr_new / rTr
        rTr = rTr_new

        for i in range(len(p)):
            p[i].data = r[i].data + beta.data * p[i].data


    return x, iter_count


if __name__ == '__main__':
    a = libSVM()
    inst = ERM(a.d, 2)
    X = Variable(a.X, requires_grad=False)
    y = Variable(a.y, requires_grad=False)

    start_time = time.time()
    for it in range(10):
        grad = inst.get_grad(X, y)
        x, iter_count = CG(inst, grad)

        for para, x_ in zip(inst.params, x):
            para.data.add_(-1., x_.data)

        g_norm = 0
        for grad_ in grad:
            g_norm += grad_.norm() ** 2
        g_norm = g_norm**(0.5)
        print("-- %i CG iters, %f NOG --" %(iter_count, g_norm.data.numpy()[0]))


    print("--- %f seconds ---" % (time.time() - start_time))

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
