############

#   @File name: BFGSetc.py
#   @Author:    Xi He
#   @Email: xih314@lehigh.edu, heeryerate@gmail.com

#   @Create date:   2017-09-20 00:03:09

#   @Last modified by:  Xi He
#   @Last Modified time:    2017-09-20 00:06:13

#   @Description:
#   @Example:

#   Don't forget to use control + option + R to re-indent

############

from .Optimizer import Optimizer

class BFGSetc(Optimizer):
    def __init__(self):
        super(BFGSetc, self).__init__()