###############

# @Author: 			   Chenxin Ma
# @Email:              machx9@gmail.com
# @Date:               2017-09-20 14:27:39
# @File Name:          ERM.py
# @Last modified by:   Heerye
# @Last modified time: 2017-09-24T16:37:16-04:00



###############

from torch.autograd import Variable
import torch
from .BasicModule import BasicModule
import sys
import os

class ERM(BasicModule):

    def __init__(self, input_dim, num_classes):
        super(ERM, self).__init__()
        self.model_name = 'ERM'

        self.linear = torch.nn.Linear(input_dim, num_classes, bias=True)
        self.linear.bias.data.zero_()
        self.linear.weight.data.zero_()

        self.params = [self.linear.weight, self.linear.bias]
        # self.params = [self.linear.weight]
        # self.model = nn.Sequential()
        # self.model.add_module(
                    # "linear",
                    # nn.Linear(input_dim, num_classes, bias=True)
                    # )
        # self.loss = torch.nn.CrossEntropyLoss(size_average=True)
        self.loss = torch.nn.CrossEntropyLoss(size_average=True)

        self.lmd = 1e-3

    def forward(self, x):

        # x = self.model(x)
        x = self.linear(x)
        return x

    def get_loss(self, x, y):

        fx = self.forward(x)
        loss = self.loss(fx, y)
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
        hv = torch.autograd.grad(gv, self.params, retain_graph=True)
        for i in range(len(hv)):
            hv[i].data.add_(self.lmd * v[i].data)
        return hv
