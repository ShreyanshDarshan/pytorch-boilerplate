import torch
import gin
import optim
import loss


@gin.configurable
class SubTrainerBase:
    def __init__(self, model, optimizer, lr):
        self.model = model
        self.optimizer = self.get_optimizer(optimizer, lr)
        self.scheduler = self.get_scheduler()
        self.loss = loss.Loss()

    def get_optimizer(self, optimizer, lr):
        optim.get_optimizer(self.model, self.optimizer, self.lr)

    def get_scheduler(self, scheduler, **kwargs):
        optim.get_scheduler(scheduler, **kwargs)

    def train_iter(self, x, y):
        self.optimizer.zero_grad()
        y_pred = self.model(x)
        loss = self.loss(y_pred, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    @torch.no_grad()
    def test_iter(self, x, y):
        y_pred = self.model(x)
        loss = self.loss(y_pred, y)
        return loss.item()
