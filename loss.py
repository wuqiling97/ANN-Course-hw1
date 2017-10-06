from __future__ import division
import numpy as np


class EuclideanLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        # input: batch_size * 10
        output = 0.5 * (input - target) ** 2
        batch_size = output.shape[0]
        return output.sum() / batch_size

    def backward(self, input, target):
        # batch_size * 10, dE/dy_i
        return (input - target) / len(target)
