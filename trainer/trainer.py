import torch
import gin
from subtrainer.subtrainer import SubTrainerBase


@gin.configurable
class Trainer:
    def __init__(
        self, model, subtrainer: SubTrainerBase, dataloaders: dict, epochs: int
    ):
        self.model = model
        self.subtrainer = subtrainer
        self.train_loader = dataloaders["train"]
        self.test_loader = dataloaders["test"]
        self.val_loader = dataloaders["val"]
        self.epochs = epochs

    def fit(self):
        for epoch in range(self.epochs):
            for x, y in self.train_loader:
                loss = self.subtrainer.train_iter(x, y)
                print(f"Epoch: {epoch}, Loss: {loss}")
