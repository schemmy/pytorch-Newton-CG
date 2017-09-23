############

#   @File name: Optimizer.py
#   @Author:    Xi He
#   @Email: xih314@lehigh.edu, heeryerate@gmail.com

#   @Create date:   2017-09-20 00:03:09

# @Last modified by:   Heerye
# @Last modified time: 2017-09-22T13:44:18-04:00

#   @Description:
#   @Example:

#   Don't forget to use control + option + R to re-indent

############

# TODO(xi): think and add necessary functions
# TODO(xi): Optimizer class refer to Optimizer class
# https://github.com/pytorch/pytorch/blob/master/torch/optim/optimizer.py


class Optimizer(object):
    """Optimizer Base."""

    def __init__(self, params, setting):
        """Init."""
        self.setting = setting  # from Settings class
        self.para_groups = list(params)

    def zero_grad(self):
        """Set gradient to zero."""
        for group in self.para_groups:
            print(type(group))
            for p in group['params']:
                if p.grad is not None:
                    p.grad.data.zero_()
