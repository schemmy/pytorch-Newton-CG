############

#   @File name: dataset.py
#   @Author:    Xi He
#   @Email: xih314@lehigh.edu, heeryerate@gmail.com

#   @Create date:   2017-09-19 21:56:19

# @Last modified by:   Heerye
# @Last modified time: 2017-09-23T17:20:39-04:00

#   @Description:
#   @Example:

#   Don't forget to use control + option + R to re-indent

############

from torch.utils import data
import torchvision
from torchvision import transforms as T
import numpy
from torch.autograd import Variable


class Mnist(data.Dataset):

    def __init__(self, root, train=True):
        '''
        Process MNIST dataset
        '''
        if not train:
            self.__name__ = 'Test'
            imgs = torchvision.datasets.MNIST(
                root=root,
                train=False
            )
            print("Loading Testing Data ...")
        else:
            self.__name__ = 'Train'
            imgs = torchvision.datasets.MNIST(
                root=root,
                train=True,
                transform=T.ToTensor(),
                download=True
            )
            print("Loading Training Data ...")

        self.imgs = imgs

    def __len__(self):
        return len(self.imgs)

class Cifra10(data.Dataset):

    def __init__(self, root, train=True):
        '''
        Process Cifra10 dataset
        '''
        if not train:
            self.__name__ = 'Test'
            imgs = torchvision.datasets.CIFAR10(
                root=root,
                train=False
            )
            print("Loading Testing Data ...")
        else:
            self.__name__ = 'Train'
            imgs = torchvision.datasets.CIFAR10(
                root=root,
                train=True,
                transform=T.ToTensor(),
                download=False
            )
            print("Loading Training Data ...")

        self.imgs = imgs

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    # MNIST dataset loading test
    # root = './cifra10/'
    root = './mnist/'
    mnist_train = Mnist(root, train=True)
    print(mnist_train.imgs.train_data.size())
    mnist_test = Mnist(root, train=False)
    print(mnist_test.imgs.test_data.size())

    import matplotlib.pyplot as plt
    idx = 12 # not too large
    length = 5 # should less than 5
    for i in range(length):

        plt.subplot(2, length, i + 1)
        plt.imshow(mnist_train.imgs.train_data[idx + i].numpy(), cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.title('%s, %i' % (mnist_train.__name__,mnist_train.imgs.train_labels[idx + i]))

        plt.subplot(2, length, length + i + 1)
        plt.imshow(mnist_test.imgs.test_data[idx + i].numpy(), cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.title('%s, %i' % (mnist_test.__name__, mnist_test.imgs.test_labels[idx + i]))

    plt.show()
