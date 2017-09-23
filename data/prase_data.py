############

#   @File name: prase_data.py
#   @Author:    Xi He
#   @Email: xih314@lehigh.edu, heeryerate@gmail.com

#   @Create date:   2017-09-19 23:28:35

# @Last modified by:   Heerye
# @Last modified time: 2017-09-22T14:27:06-04:00

#   @Description:
#   @Example:

#   Don't forget to use control + option + R to re-indent

############


import os
import urllib.request
import numpy as np
import torch
from scipy.sparse import csr_matrix, vstack


labelMap = {
    '-1': 0,
    '+1':  1
}


class libSVM():

    def __init__(self, name='a1a', train=True):

        if not os.path.exists('files/'):
            os.makedirs('files')

        file_path = "files/%s" % (name)
        if not os.path.exists(file_path):
            link_name = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/%s' % (
                name)
            urllib.request.urlretrieve(link_name, file_path)

        self.load_csr(file_path)

    def load_csr(self, file_path):
        label = []
        with open(file_path, 'r') as f:
            col_idx = []
            d = []
            indptr = [0]
            for line in f:
                o = line.split(" ")
                label.append(labelMap[o[0]])

                for word in o[1: -1]:
                    key = word.split(":")
                    col_idx.append(int(key[0]) - 1)
                    d.append(float(key[1]))
                indptr.append(len(d))

        self.X = csr_matrix((d, col_idx, indptr))
        self.y = np.array(label)
        self.n, self.d = self.X.shape

        self.X = torch.from_numpy(self.X.todense()).type(torch.FloatTensor)
        self.y = torch.from_numpy(self.y).type(torch.LongTensor)

# a = libSVM()
# print a.d
