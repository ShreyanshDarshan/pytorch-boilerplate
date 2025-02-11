import torch.optim as optim
import gin

def get_optimizer(model, optimizer, lr):
    return optimizer(model.parameters(), lr=lr)

def get_scheduler(optimizer, scheduler, **kwargs):
    return scheduler(optimizer, **kwargs)
    
