############

#   @File name: main.py
#   @Author:    Xi He
#   @Email: xih314@lehigh.edu, heeryerate@gmail.com

#   @Create date:   2017-09-19 21:53:18

# @Last modified by:   Heerye
# @Last modified time: 2017-09-22T08:16:44-04:00

#   @Description:
#   @Example:

#   Don't forget to use control + option + R to re-indent
#   Run python3 -m visdom.server
#   open http://localhost:8097 in browser

############

import os

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchnet import meter

from settings import setting

import models
from data import Mnist

import optimizer


# TODO(xi): move to unitest later
def dataLoadingTest(**kwargs):
    setting.parse(kwargs)
    train_data = Mnist(setting.train_data_root, train=True)
    test_data = Mnist(setting.test_data_root, train=False)
    print(train_data.imgs.train_data.size())
    print(test_data.imgs.test_data.size())


def train(**kwargs):
    setting.parse(kwargs)
    model = getattr(models, setting.model)()
    if setting.load_model_path:
        model.load(setting.load_model_path)
    if setting.use_gpu:
        model.cuda()

    train_data = Mnist(setting.train_data_root, train=True)
    test_data = Mnist(setting.train_data_root, train=False)

    train_loader = DataLoader(train_data.imgs, setting.batch_size,
                              shuffle=True, num_workers=setting.num_workers)
    test_loader = DataLoader(test_data.imgs, setting.batch_size,
                             shuffle=False, num_workers=setting.num_workers)

    criterion = torch.nn.CrossEntropyLoss()

    # TODO(xi): user define the optimizer in Optimizer class
    optimizer = torch.optim.Adam(
        model.parameters(), lr=setting._lr, weight_decay=setting.weight_decay)
    # optim = optimizer.FirstOrder(model.parameters(), setting=setting)

    loss_meter = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(setting.num_classes)
    previous_loss = 1e100

    loss_list = []
    jump_point = []
    for epoch in range(setting.max_epoch):

        loss_meter.reset()
        confusion_matrix.reset()

        train_loader = DataLoader(train_data.imgs, setting.batch_size*(epoch+1),
                              shuffle=False, num_workers=setting.num_workers)

        previous_loss = 0.0

        for iter, (data, label) in enumerate(train_loader):

            input = Variable(data)
            target = Variable(label)

            if setting.use_gpu:
                input = input.cuda()
                target = target.cuda()

            optimizer.zero_grad()
            score = model(input)
            loss = criterion(score, target)
            loss.backward()
            optimizer.step()

            loss_meter.add(loss.data[0])
            confusion_matrix.add(score.data, target.data)

            if iter == 1:
                previous_loss = loss_meter.value()[0]

            if iter % setting.print_freq == setting.print_freq - 1:
                print('epoch: %i, iter: %i, loss: %f' %
                      (epoch, iter, loss_meter.value()[0]))

                if setting.do_debug and os.path.exists(setting.debug_file):
                    import ipdb
                    ipdb.set_trace()

                print(loss_meter.value()[0], previous_loss)

                loss_list.append(loss_meter.value()[0])

                if loss_meter.value()[0] <= 0.9*previous_loss:
                    jump_point.append(loss_meter.value()[0])
                    print(loss_meter.value()[0], previous_loss)
                    print("starting new adam.")
                    break


        model.save()

    import matplotlib.pyplot as plt
    plt.plot(loss_list)
    plt.plot(jump_point)
    plt.show()

        # test_cm, test_accuracy = test(model, test_loader)

        # print("epoch:{epoch},lr:{lr},loss:{loss},train_cm:{train_cm},test_cm:{test_cm}".format(
        #     epoch=epoch, loss=loss_meter.value()[0], test_cm=str(test_cm.value()), train_cm=str(confusion_matrix.value()), lr=setting.lr))


def write_csv(results, file_name):
    import csv
    with open(file_name, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'label'])
        writer.writerows(results)


def help():
    print('''
    usage : python file.py <function> [--args=value]
    <function> := train | test | help
    example:
            python {0} train --lr=0.01
            python {0} test --dataset='path/to/dataset/root/'
            python {0} help
    avaiable args:'''.format(__file__))

    from inspect import getsource
    source = (getsource(setting.__class__))
    print(source)


if __name__ == '__main__':
    import fire
    fire.Fire()
