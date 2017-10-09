from result import *
from plot import *
import matplotlib.pyplot as plt
import numpy as np


max_iter = np.argmax(test_acc)
print('max accuracy at iteration {}, value = {}'.format(test_iter[max_iter], test_acc[max_iter]))
plot_loss(train_iter, train_loss, 'Training Loss')
plot_loss(test_iter, test_loss)
plot_acc(test_iter, test_acc)