############

#   @File name: NaiveCNet.py
#   @Author:    Xi He
#   @Email: xih314@lehigh.edu, heeryerate@gmail.com

#   @Create date:   2017-09-19 23:32:11

# @Last modified by:   Heerye
# @Last modified time: 2017-09-25T16:25:00-04:00

#   @Description:
#   @Example:

#   Don't forget to use control + option + R to re-indent

############

import torch
from torch import nn
try:
    from BasicModule import BasicModule
except:
    from .BasicModule import BasicModule
from torch.autograd import gradcheck

torch.manual_seed(123)

class NaiveCNet(BasicModule):
    """Naive convolutional neural network for MNIST dataset classification."""

    def __init__(self, num_classes=10):
        """Init."""
        super(NaiveCNet, self).__init__()
        self.model_name = 'NaiveCNet'

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 7 * 7, num_classes))

        # self.softmax = nn.Sequential(nn.Softmax())

        self.lmd = 1e-3

    def get_lossfn(self):
        return torch.nn.CrossEntropyLoss(size_average=True)

    def get_named_params(self):

        weights, biases = [], []
        for name, p in self.named_parameters():
            if 'bias' in name:
                biases += [p]
            else:
                weights += [p]
        return [{'params': weights}, {'params': biases}]

    def get_params(self):
        return self.parameters()

    def forward(self, x):
        """Forward."""
        # print("in forward: %.4f"%x.data.sum())
        x = self.features(x)
        # x = x.view(x.size(0), -1)
        # x = self.classifier(x)
        # x = self.softmax(x)
        # print("in forward: %.4f"%x.data.sum())

        return x

    def get_loss(self, x, y):

        fx = self.forward(x)
        loss = self.get_lossfn()(fx, y)

        return loss

    def get_grad(self, x, y):

        params = self.get_params()
        loss = self.get_loss(x, y)
        g = torch.autograd.grad(loss, params, create_graph=True)
        # for i in range(len(g)):
        #     g[i].data.add_(self.lmd * params[i].data)
        return g

    def get_Hv(self, grad, v):

        params = self.get_params()

        gv = 0.0
        for g_para, v_para in zip(grad, v):
            gv += (g_para * v_para).sum()
        hv = torch.autograd.grad(gv, params, retain_graph=True)

        # for i in range(len(hv)):
        #     hv[i].data.add_(self.lmd * v[i].data)

        return hv

    def grad_test(self, x, y):

        params = self.get_params()

        loss = self.get_loss(x, y)

        grad = self.get_grad(x, y)

        eps = 1e-16
        for para in params:
            para.data.add_(torch.ones(para.size()) * eps)

        loss_y = self.get_loss(x, y)

        g = 0.0
        for para in grad:
            g += para.sum()

        print("eps: %e, finite_diff - g: %.16e" %
              (eps, (loss_y.data[0] - loss_y.data[0]) - eps * g.data[0]))


if __name__ == '__main__':
    import sys
    import os
    sys.path.append(os.path.abspath('..'))
    import models

    from settings import setting
    from data.dataset import Mnist
    model = getattr(models, setting.model)()
    net = NaiveCNet()

    setting.train_data_root = "../data/mnist/"

    train_data = Mnist(setting.train_data_root, train=True)
    test_data = Mnist(setting.train_data_root, train=False)

    import torch
    from torch.utils.data import DataLoader
    from torch.autograd import Variable

    train_loader = DataLoader(
        train_data.imgs, setting.batch_size, shuffle=False, num_workers=setting.num_workers)
    # test_loader = DataLoader(test_data.imgs, setting.batch_size,
    #                          shuffle=False, num_workers=setting.num_workers)

    test_x = Variable(torch.unsqueeze(test_data.imgs.test_data, dim=1), volatile=True).type(torch.FloatTensor)[
        :2000] / 255.   # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
    test_y = test_data.imgs.test_labels[:2000]

    loss_fn = net.get_lossfn()

    # print(net.__call__)

    ######################################## TEST 1 #########
    # for iter, (data, label) in enumerate(train_loader):
    #     input = Variable(data)
    #     target = Variable(label)
    #
    #     # TODO(xi): check how to use the gradcheck fuction to verify get_grad
    #     # gradcheck(net.get_loss, [input, target])
    #     # net.grad_test(input, target)
    #
    #     out = model(input)
    #     loss = net.get_loss(input, target)
    #
    #     # print(out, loss)
    #
    #     g = net.get_grad(input, target)
    #
    #     # print(g)
    #
    #     v = []
    #     for para in net.get_params():
    #         v.append(Variable(torch.ones(para.size())))
    #
    #     Hv = net.get_Hv(g, v)
    #     print(Hv)
    #
    #     break

    # ######################################## TEST 2 #########
    optimizer = torch.optim.Adam(
        model.parameters(), lr=setting._lr, weight_decay=setting.weight_decay)
    #
    import time
    start = time.time()
    for iter, (data, label) in enumerate(train_loader):

        input = Variable(data)
        target = Variable(label)

        print(input.sum())

        optimizer.zero_grad()
        # TODO(xi): figure out what's the difference between model(input), model.forward(input)
        out = model(input)
        # out = net.forward(input)
        # out = net.features(input)


        print(out.sum())

        break
    #
    #     # TODO(xi): figure out what's the difference between net.get_lossfn()(out, target) and net.get_loss(input, target)
    #     # loss = net.get_lossfn()(out, target)
    #     # loss = net.get_loss(input, target)
    #     # loss.backward()
    #     # optimizer.step()
    #     #
    #     # if iter % setting.print_freq == 0:
    #     #     test_output = model(test_x)
    #     #     pred_y = torch.max(test_output, 1)[1].data.squeeze()
    #     #     accuracy = sum(pred_y == test_y) / float(test_y.size(0))
    #     #
    #     #     print("iter: %d, loss: %.4f, accuracy: %.4f, time: %.4f" %
    #     #           (iter, loss.data[0], accuracy, time.time() - start))

    ####################################### TEST 3 #########
    # from testfiles_cx.demo_cg_xi import CG
    # import time
    # start_time = time.time()
    # for iter, (data, label) in enumerate(train_loader):
    #     input = Variable(data)
    #     target = Variable(label)
    #     grad = net.get_grad(input, target)
    #     x, iter_count = CG(net, grad)
    #
    #     for para, x_ in zip(net.get_params(), x):
    #         # print(para.size(), x_.size())
    #         para.data.add_(-1., x_.data)
    #
    #     g_norm = 0.0
    #     for grad_ in grad:
    #         g_norm += grad_.norm()**2
    #     g_norm = g_norm**(0.5)
    #     print("-- %i CG iters, %f NOG --" %(iter_count, g_norm.data.numpy()[0]))
    #     # break
