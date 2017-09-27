############

#   @File name: NaiveCNet.py
#   @Author:    Xi He
#   @Email: xih314@lehigh.edu, heeryerate@gmail.com

#   @Create date:   2017-09-19 23:32:11

# @Last modified by:   Heerye
# @Last modified time: 2017-09-26T18:01:48-04:00

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
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        # x = self.softmax(x)
        # print("execute forward in forward function!!")

        return x

    def get_loss(self, x, y):

        fx = self.forward(x)
        loss = self.get_lossfn()(fx, y)

        # print("execute forward in get_loss function!!")

        return loss

    def get_grad(self, x, y):

        params = self.get_params()
        loss = self.get_loss(x, y)
        g = torch.autograd.grad(loss, params, create_graph=True)
        # print("execute forward in get_grad function!!")
        # for i in range(len(g)):
        #     g[i].data.add_(self.lmd * params[i].data)
        return g

    def get_Hv(self, grad, v):

        params = self.get_params()

        gv = 0.0
        for g_para, v_para in zip(grad, v):
            gv += (g_para * v_para).sum()
        hv = torch.autograd.grad(gv, params, create_graph=True)

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


def line_search(lr, net, direction, X, y, grad):

    w = []
    for para in net.get_params():
        w.append(para.clone())

    f0 = net.get_loss(X, y)
    f_new = f0+1.

    gd = 0.0
    for g, d in zip(grad, direction):
        gd += (g * d).sum()

    if gd.data[0] <= 0.0:
        lr = -lr

    ls_it = 0
    max_ls = 5
    while (f_new >= f0).data[0] and ls_it < max_ls:

        for para, w_, d_ in zip(net.get_params(), w, direction):
            para.data = w_.data.add(-lr, d_.data)

        f_new = net.get_loss(X, y)
        lr *=0.9
        ls_it += 1

    if ls_it == max_ls:
        for para, w_, g in zip(net.get_params(), w, grad):
            para.data = w_.data.add(-0.01, g.data)

        # print (f_new.data[0], f0.data[0])

    return f_new, lr, ls_it

if __name__ == '__main__':
    import sys
    import os
    sys.path.append(os.path.abspath('..'))
    import models

    from settings import setting
    from data.dataset import Mnist

    setting.model = "NaiveCNet"
    net = getattr(models, setting.model)()

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
    # optimizer = torch.optim.Adam(
    #     net.parameters(), lr=setting._lr, weight_decay=setting.weight_decay)
    #
    # import time
    # start = time.time()
    # for iter, (data, label) in enumerate(train_loader):
    #
    #     input = Variable(data)
    #     target = Variable(label)
    #
    #     optimizer.zero_grad()
    #
    #     # out = net.forward(input)
    #
    #     loss = net.get_loss(input, target)
    #     loss.backward()
    #     optimizer.step()
    #
    #     if iter % setting.print_freq == 0:
    #         test_output = net(test_x)
    #         pred_y = torch.max(test_output, 1)[1].data.squeeze()
    #         accuracy = sum(pred_y == test_y) / float(test_y.size(0))
    #
    #         print("iter: %d, loss: %.4f, accuracy: %.4f, time: %.4f" %
    #               (iter, loss.data[0], accuracy, time.time() - start))


    ####################################### TEST 3 #########
    from testfiles_cx.demo_cg_xi import CG
    import time
    start_time = time.time()
    lst_nog, lst_loss, lst_acc = [], [], []
    for iter, (data, label) in enumerate(train_loader):
        X = Variable(data)
        y = Variable(label)

        ##########  first-order methods ############
        # grad = net.get_grad(X, y)
        # lr = 0.1
        # for para, g in zip(net.get_params(), grad):
        #     para.data.add_(-lr * g.data)
        #
        #
        # if iter % setting.print_freq == 1:
        #     g_norm = 0.0
        #     for grad_ in grad:
        #         g_norm += grad_.norm()**2
        #     g_norm = g_norm**(0.5)
        #
        #     loss = net.get_loss(X, y)
        #     # print("get loss")
        #
        #     test_output = net(test_x)
        #     # print("get test output")
        #     pred_y = torch.max(test_output, 1)[1].data.squeeze()
        #     accuracy = sum(pred_y == test_y) / float(test_y.size(0))
        #
        #     print("-- %i iters, %f NOG, %f loss, %.4f acc --" \
        #             %(iter, g_norm.data.numpy()[0], loss.data[0], accuracy))
        #
        # # break

        for i in range(50):
            grad = net.get_grad(X, y)

            direction, iter_count = CG(net, grad)

            lr = 1.0
            loss, lr, ls_it = line_search(lr, net, direction, X, y, grad)

            g_norm = 0.0
            for grad_ in grad:
                g_norm += grad_.norm()**2
            g_norm = g_norm**(0.5)

            test_output = net(test_x)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = sum(pred_y == test_y) / float(test_y.size(0))

            lst_nog.append(g_norm.data.numpy()[0])
            lst_loss.append(loss.data[0])
            lst_acc.append(accuracy)
            print("-- %i CG iters, %f NOG, %f loss, %i ls, %.4f acc --" \
                    %(iter_count, g_norm.data.numpy()[0], loss.data[0], ls_it, accuracy))

    import matplotlib.pyplot as plt
    fig=plt.figure(1, figsize=(8, 4*(numpy.sqrt(5)-1)))
    fig.subplots_adjust(bottom=0.1, left=0.12)
    plt.style.use('fivethirtyeight')
    plt.plot(lst_loss, label='loss')
    plt.plot(lst_nog, label='NOG')
    plt.plot(lst_acc, label='Acc')
    plt.title("MNIST, Conv")
    plt.xlabel("Iterations")
    plt.ylabel("Batch Loss Value")
    plt.legend()
    # plt.show()
    plt.savefig('naive_newton_cg.png')
