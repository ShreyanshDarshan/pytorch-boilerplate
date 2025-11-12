import gin
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

@gin.configurable
class MLP(nn.Module):
    def __init__(self,
                 device: torch.device,
                 in_channels: int, 
                 out_channels: int, 
                 hidden_channels: int = 128, 
                 num_layers: int = 3):
        super(MLP, self).__init__()
        layers = []
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else hidden_channels
            out_ch = out_channels if i == num_layers - 1 else hidden_channels
            layers.append(nn.Linear(in_ch, out_ch))
            if i != num_layers - 1:
                layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)

    def forward(self, data):
        x = data['inputs']  # [B, in_channels]

        out = {
            'output': self.mlp(x)  # [B, out_channels]
        }

        return out


if __name__ == '__main__':
    model = MLP(in_channels=3, out_channels=3, hidden_channels=64, num_layers=4)
    model = model.cuda()
    model.train()
    print(model)
    input = torch.randn(2, 10).cuda()

    out = model(input)
    breakpoint()
