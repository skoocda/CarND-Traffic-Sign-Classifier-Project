'''
Utility functions for training on the German sign set
'''

import os
import sys
import time
import random
import warnings
import numpy as np
import matplotlib
#from model import Paths
from collections import namedtuple
from matplotlib import pyplot

#print(sys.version)

#def plot_curve(axis, params, train_column, valid_column, linewidth=2, train_linestyle='b-', valid_linestyle='g-'):
#    model_history = np.load(Paths(params).train_history_path + '.npz')
#    train_values = model_history[train_column]
#    valid_values = model_history[valid_column]
#    epochs = train_values.shape(0)
#    x_axis = np.arange(epochs)
#    axis.plot(x_axis[train_values > 0], train_values[train_values > 0], train_linestyle, linewidth=linewidth, label='train')
#    axis.plot(x_axis[valid_values > 0], valid_values[valid_values > 0], valid_linestyle, linewidth=linewidth, label='valid')
#    return epochs
#
#def plot_learning_curves(params):
#    curves_figure = pyplot.figure(figsize=(10, 4))
#    axis = curves_figure.add_subplot(1, 2, 1)
#    epochs_plotted = plot_curve(axis, params, train_column='train_accuracy_history', valid_column='valid_accuracy_history')
#    pyplot.grid()
#    pyplot.legend()
#    pyplot.xlabel('epoch')
#    pyplot.ylabel('accuracy')
#    pyplot.xlim(0, epochs_plotted)
#    pyplot.ylim(50., 115.)
#    axis = curves_figure.add_subplot(1, 2, 2)
#    epochs_plotted = plot_curve(axis, params, train_column='train_loss_history', valid_column='valid_loss_history')
#    pyplot.grid()
#    pyplot.legend()
#    pyplot.xlabel('epoch')
#    pyplot.ylabel('loss')
#    pyplot.xlim(0, epochs_plotted)
#    pyplot.ylim(0.0001, 10.)
#    pyplot.yscale('log')
#    pyplot.show()

def get_time_hhmmss(start=None):
    if start is None:
        return time.strftime("%Y/%m/%d %H:%M:%S")
    end = time.time()
    m, s = divmod(end-start, 60)
    h, m = divmod(m, 60)
    time_str = "%02d:%02d:%02d" % (h, m, s)
    return time_str

def print_progress(iteration, total):
    str_format = "{0:.0f}"
    percents = str_format.format(100*(iteration/float(total)))
    filled_length = int(round(100*iteration/float(total)))
    bar = '▮'* filled_length + '-' * (100-filled_length)
    sys.stdout.write('\r |%s| %s%%' % (bar, percents))
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()

Parameters = namedtuple('Parameters', [
    'num_classes',
    'image_size', 'batch_size',
    'max_epochs', 'log_epoch', 'print_epoch',
    'learning_rate_decay', 'learning_rate',
    'l2_reg_enabled', 'l2_lambda',
    'early_stopping_enabled', 'early_stopping_patience',
    'resume_training',
    'conv1_k', 'conv1_d', 'conv1_p',
    'conv2_k', 'conv2_d', 'conv2_p',
    'conv3_k', 'conv3_d', 'conv3_p',
    'fc4_size', 'fc4_p'
])

