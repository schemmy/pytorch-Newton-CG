##################

# @Author: 			   Chenxin Ma
# @Email: 			   machx9@gmail.com
# @Date:               2017-09-22 23:32:45
# @Last Modified by:   Chenxin Ma
# @Last Modified time: 2017-09-23 00:51:20

##################

import numpy as np
import torch


def get_flat_params_from(parameters):
    params = []
    for para in parameters:
        params.append(para.data.view(-1))

    flat_params = torch.cat(params)
    return flat_params


def set_flat_params_to(model, flat_params):
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(
            flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size