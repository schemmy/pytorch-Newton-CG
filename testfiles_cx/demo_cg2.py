##################

# @Author: 			   Chenxin Ma
# @Email: 			   machx9@gmail.com
# @Date:               2017-09-23 00:41:49
# @Last Modified by:   Chenxin Ma
# @Last Modified time: 2017-09-24 10:23:51

##################



import sys
import os
sys.path.append(os.path.abspath('..'))
from data.dataset import Mnist
from data.prase_data import libSVM
from models.ERM import ERM
from utils.transform import *

from torch.autograd import Variable


a = libSVM()
inst = ERM(a.d, 2)
X = Variable(a.X, requires_grad=False)
y = Variable(a.y, requires_grad=False)
grad = inst.get_grad(X, y)

v = get_flat_params_from(inst.params)
Ax = inst.get_Hv(grad, v)
Ax = get_flat_params_from(Ax)

r = grad_flat - Ax
p = r.clone()

rTr = torch.dot(r, r)

for i in range(10):
	Ap = inst.get_Hv(grad, p)
