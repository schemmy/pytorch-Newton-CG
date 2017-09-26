############

#   @File name: BasicModule.py
#   @Author:    Xi He
#   @Email: xih314@lehigh.edu, heeryerate@gmail.com

#   @Create date:   2017-09-19 23:32:11

# @Last modified by:   Heerye
# @Last modified time: 2017-09-24T17:18:36-04:00

#   @Description:
#   @Example:

#   Don't forget to use control + option + R to re-indent

############

import torch
import time


class BasicModule(torch.nn.Module):

    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def save(self, name=None):
        if name is None:
            prefix = 'checkpoints/' + self.model_name + '_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        torch.save(self.state_dict(), name)
        return name

    # def forward(self, x):
    #     pass
    #
    # def loss(self, out, y):
    #     pass
    #
    # def get_loss(self, x, y, lmd):
    #
    #     out = self.forward(x)
    #     loss = self.loss(out, y)
    #
    #     return loss

    # def get_grad(self, x, y):
    #
    #     out = self.get_loss(x, y)
    #     g = torch.autograd.grad(out, self.params, create_graph=True)
    #     for i in range(len(g)):
    #         g[i].data.add_(self.lmd * self.params[i].data)
    #     return g
    #
    # def get_Hv(self, grad, v):
    #     gv = 0
    #     for g_para, v_para in zip(grad, v):
    #         gv += (g_para * v_para).sum()
    #     hv = torch.autograd.grad(gv, self.params, retain_graph=True)
    #     for i in range(len(hv)):
    #         hv[i].data.add_(self.lmd * v[i].data)
    #     return hv

# class Flat(torch.nn.Module):

#     def __init__(self):
#         super(Flat, self).__init__()

#     def forward(self, x):
#         return x.view(x.size(0), -1)
