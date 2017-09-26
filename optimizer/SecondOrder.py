############

#   @File name: SecondOrder.py
#   @Author:    Xi He
#   @Email: xih314@lehigh.edu, heeryerate@gmail.com

#   @Create date:   2017-09-20 00:03:09

# @Last modified by:   Heerye
# @Last modified time: 2017-09-25T13:33:32-04:00

#   @Description:
#   @Example:

#   Don't forget to use control + option + R to re-indent

############

from .Optimizer import Optimizer

class SecondOrder(Optimizer):

    def __init__(self, params, setting):
        super(SecondOrder, self).__init__()

    def NewtonCG(self, lr=1.):
        pass
    	# step size is fixed to be 1.0
