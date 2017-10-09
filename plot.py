import matplotlib.pyplot as plt
import numpy as np


def __plot(x, y, xlabel, ylabel, title):
    plt.plot(x, y, '-')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


def plot_loss(x, title=''):
    plt.plot(x, y, '-')
    plt.xlabel('iterations over epoch')
    plt.ylabel('loss')
    plt.title(title)
    plt.ylim(ymin=0)
    plt.show()


def plot_2loss(xtrain, ytrain, xtest, ytest, title=''):
    plt.plot(xtrain, ytrain, '-', label='Training')
    plt.plot(xtest, ytest, '-', label='Testing')
    plt.xlabel('iterations over epoch')
    plt.ylabel('loss')
    plt.legend(loc='upper center', framealpha=0.5, ncol=3)
    plt.title(title)
    plt.ylim(ymin=0)
    plt.show()


def plot_acc(x, y, title=''):
    plt.plot(x, y, '.-')
    plt.xlabel('iterations over epoch')
    plt.ylabel('accuracy')
    plt.title(title)
    plt.ylim(ymax=1)
    plt.show()
