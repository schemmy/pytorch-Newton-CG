##################

# @Author: 			   Chenxin Ma
# @Email: 			   machx9@gmail.com
# @Date:               2017-09-23 00:41:49
# @Last Modified by:   Chenxin Ma
# @Last Modified time: 2017-09-23 00:45:49

##################



import sys
import os
sys.path.append(os.path.abspath('..'))
from data.dataset import Mnist
from data.prase_data import libSVM
from models.ERM import ERM
from utils.transform import *

from torch.autograd import Variable

