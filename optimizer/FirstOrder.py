############

#   @File name: FirstOrder.py
#   @Author:    Xi He
#   @Email: xih314@lehigh.edu, heeryerate@gmail.com

#   @Create date:   2017-09-20 00:03:09

# @Last modified by:   Heerye
# @Last modified time: 2017-09-22T08:08:22-04:00

#   @Description:
#   @Example:

#   Don't forget to use control + option + R to re-indent

############

from .Optimizer import Optimizer


class FirstOrder(Optimizer):
    """Set of FirstOrder Optimizer.

    Example:
        optimizer=FirstOrder(model.parameters(), setting)
        optimizer.zero_grad()
        loss_fn(model(input), target).backward()
        optimizer.sgd(lr=0.1)
    """

    def __init__(self, params, setting):
        """Initilizer."""
        super(FirstOrder, self).__init__(params, setting)

    def sgd(self, lr=None):
        """Perform a single sgd step."""
        if lr is None:
            lr = self.setting._lr
        for group in self.para_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                p.data.add_(lr, p.grad.data)

    def heavyBall(self, lr=None, mom=None):
        """Perform a single heavyBall step."""
        if lr is None:
            Warning("Default learning rate is used.")
            lr = self.setting._lr
        if mom is None:
            Warning("Default heavyball momentum parameter is used.")
            mom = self.setting._mom
        if mom <= 0:
            raise ValueError(
                "heavyball momentum parameter should be positive.")

        momentum_buffer = None
        # FIXME(xi): may need involve a global bool variable for momentum state
        for group in self.para_groups():
            for p in group['params']:
                if momentum_buffer is not None:
                    buf = p.clone()
                else:
                    buf = momentum_buffer
                    buf.mul_(mom).add_(p)

                p.data.add_(lr, buf)
