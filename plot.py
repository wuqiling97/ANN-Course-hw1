import matplotlib.pyplot as plt
import numpy as np


def __plot(x, y, xlabel, ylabel, title):
    plt.plot(x, y, '-')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


def plot_loss(x, y, title=''):
    plt.plot(x, y, '-')
    plt.xlabel('iteration over epoch')
    plt.ylabel('loss')
    plt.title(title)
    plt.ylim(ymin=0)
    plt.show()


def plot_acc(x, y, title=''):
    plt.plot(x, y, '.-')
    plt.xlabel('iteration over epoch')
    plt.ylabel('accuracy')
    plt.title(title)
    plt.ylim(ymax=1)
    plt.show()
