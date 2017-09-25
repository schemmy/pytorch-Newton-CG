############

#   @File name: SecondOrder.py
#   @Author:    Xi He
#   @Email: xih314@lehigh.edu, heeryerate@gmail.com

#   @Create date:   2017-09-20 00:03:09

#   @Last modified by:  Xi He
#   @Last Modified time:    2017-09-20 00:05:59

#   @Description:
#   @Example:

#   Don't forget to use control + option + R to re-indent

############

from .Optimizer import Optimizer

class SecondOrder(Optimizer):

    def __init__(self, params, setting):
        super(SecondOrder, self).__init__()

    def Newton(self, lr=1.):
    	# step size is fixed to be 1.0