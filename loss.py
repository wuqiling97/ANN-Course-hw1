from __future__ import division
import numpy as np


class EuclideanLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        output = (input - target) ** 2
        output = 0.5 * np.mean(output, axis=1)
        return output

    def backward(self, input, target):
        return input - target
