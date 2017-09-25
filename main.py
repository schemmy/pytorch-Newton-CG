############

#   @File name: main.py
#   @Author:    Xi He
#   @Email: xih314@lehigh.edu, heeryerate@gmail.com

#   @Create date:   2017-09-19 21:53:18

# @Last modified by:   Heerye
# @Last modified time: 2017-09-23T18:30:23-04:00

#   @Description:
#   @Example:

#   Don't forget to use control + option + R to re-indent
#   Run python3 -m visdom.server
#   open http://localhost:8097 in browser

############

import os
import numpy

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
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
    subset_sampler = SubsetRandomSampler(indices=range(setting.batch_size))
    train_loader = DataLoader(train_data.imgs, setting.batch_size,
                              shuffle=False, sampler=subset_sampler, num_workers=4)
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

    train_loader = DataLoader(
        train_data.imgs, setting.batch_size, shuffle=True, num_workers=setting.num_workers)
    test_loader = DataLoader(test_data.imgs, setting.batch_size,
                             shuffle=False, num_workers=setting.num_workers)

    loss_fn = torch.nn.CrossEntropyLoss()

    # TODO(xi): user define the optimizer in Optimizer class
    # optimizer = torch.optim.Adam(
    #     model.parameters(), lr=setting._lr, weight_decay=setting.weight_decay)
    # optim = optimizer.FirstOrder(model.parameters(), setting=setting)

    previous_loss = 1e100

    loss_list = []
    jump_point = []

    batch_multiplier = 1
    jump_start = 0
    for iter in range(setting._max_iters):

        batch_multiplier = iter + 1

        subset_sampler = SubsetRandomSampler(indices=range(batch_multiplier * setting.batch_size))
        train_loader = DataLoader(train_data.imgs, setting.batch_size * batch_multiplier,
                                      shuffle=False, sampler=subset_sampler, num_workers=4)
        optimizer = torch.optim.Adam(
                model.parameters(), lr=setting._lr/batch_multiplier, weight_decay=setting.weight_decay)

        tag = True
        while True:
            for batch_id, (data, label) in enumerate(train_loader):
                input = Variable(data)
                target = Variable(label)

                if setting.use_gpu:
                    input = input.cuda()
                    target = target.cuda()

                optimizer.zero_grad()
                score = model(input)
                loss = loss_fn(score, target)
                loss.backward()
                optimizer.step()

                if tag == True:
                    previous_loss = loss.data[0]
                    tag = False
                current_loss = loss.data[0]
                loss_list.append(current_loss)
                jump_start += 1
                print(batch_id, previous_loss, current_loss)
            if current_loss <= 0.5 * previous_loss:
                jump_point.append(jump_start)
                break

        print('iter: %i, loss: %f' %(iter, loss.data[0]))

    # model.save()

    import matplotlib.pyplot as plt
    fig=plt.figure(1, figsize=(8, 4*(numpy.sqrt(5)-1)))
    fig.subplots_adjust(bottom=0.1, left=0.12)
    plt.style.use('fivethirtyeight')
    plt.plot(loss_list, label='Adam')
    for id, i in enumerate(jump_point):
        if id == 0:
            plt.axvline(i, color='r', linestyle='-.', label='Restart')
        else:
            plt.axvline(i, color='r', linestyle='-.')
    plt.title("MNIST, Conv")
    plt.xlabel("Iterations")
    plt.ylabel("Batch Loss Value")
    plt.legend()
    # plt.show()
    plt.savefig('ada_adam.png')


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
