############

#   @File name: NaiveCNet.py
#   @Author:    Xi He
#   @Email: xih314@lehigh.edu, heeryerate@gmail.com

#   @Create date:   2017-09-19 23:32:11

# @Last modified by:   Heerye
# @Last modified time: 2017-09-22T13:46:19-04:00

#   @Description:
#   @Example:

#   Don't forget to use control + option + R to re-indent

############

from torch import nn
from .BasicModule import BasicModule


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
            nn.Linear(32 * 7 * 7, num_classes),
        )

    def forward(self, x):
        """Forward."""
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
