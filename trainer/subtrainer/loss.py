import torch
import gin

@gin.configurable
class Loss:
    def __init__(self, criterion):
        self.criterion = criterion

    def __call__(self, y_pred, y_true):
        return self.criterion(y_pred, y_true)