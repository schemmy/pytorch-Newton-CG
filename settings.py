############

#   @File name: settings.py
#   @Author:    Xi He
#   @Email: xih314@lehigh.edu, heeryerate@gmail.com

#   @Create date:   2017-09-19 21:53:14

# @Last modified by:   Heerye
# @Last modified time: 2017-09-22T08:11:10-04:00

#   @Description:
#   @Example:

#   Don't forget to use control + option + R to re-indent

############

import warnings

class DefaultSettings(object):

    train_data_root = './data/mnist/'
    test_data_root = './data/mnist/'

    env = 'default'
    model = 'NaiveCNet'
    load_model_path = None

    use_gpu = False

    batch_size = 8
    num_workers = 8

    max_epoch = 3
    _lr = 0.1
    _mom = 0.9
    weight_decay = 1e-4
    num_classes = 10

    print_freq = 20

    do_debug = False
    debug_file = './tmp/debug'
    result_file = 'res.csv'

def parse(self, kwargs):
    for k, v in kwargs.items():
        if not hasattr(self, k):
            warnings.warn("Warning: setting %s does not exist." %k)
        setattr(self, k, v)

    print('user setting:')
    for k, v in self.__class__.__dict__.items():
        if not k.startswith('__') and k is not 'parse':
            print(k, getattr(self, k))

DefaultSettings.parse = parse
setting = DefaultSettings()
