############

#   @File name: BasicModule.py
#   @Author:    Xi He
#   @Email: xih314@lehigh.edu, heeryerate@gmail.com

#   @Create date:   2017-09-19 23:32:11

#   @Last modified by:  Xi He
#   @Last Modified time:    2017-09-19 23:51:13

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


# class Flat(torch.nn.Module):

#     def __init__(self):
#         super(Flat, self).__init__()

#     def forward(self, x):
#         return x.view(x.size(0), -1)
