from io import BytesIO
import torch
import numpy as np
import torch.nn.functional as F
import cv2
from .base import BaseSubtrainer
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import gin
import open3d as o3d
from utils.utils import CosineWarmupScheduler
from typing import Tuple
from PIL import Image
import matplotlib.pyplot as plt

@gin.configurable()
class PolynomialSubtrainer(BaseSubtrainer):
    def __init__(self, device):
        super().__init__(device)

    def compute_loss(self, data: dict, out: dict) -> Tuple[dict, bool]:
        loss_dict = {}
        invalid_data = False
        
        pred = out['output']  # [B, 1]
        gt = data['output']  # [B, 1]    

        loss_dict['mse_loss'] = F.mse_loss(pred, gt)

        loss_dict['total_loss'] = 0
        for k, v in loss_dict.items():
            if k == 'total_loss':
                continue
            if not torch.isnan(v).any():
                loss_dict['total_loss'] = loss_dict['total_loss'] + v

        return loss_dict, invalid_data
    
    def eval_out(self, step: int, data: dict, out: dict) -> dict:
        x = data['inputs'][:, 0:1]  # [B, 1]
        y = data['inputs'][:, 1:2]  # [B, 1]
        gt = data['output']  # [B, 1]
        pred = out['output']  # [B, 1]

        x = x.detach().cpu().numpy().flatten()
        y = y.detach().cpu().numpy().flatten()
        gt = gt.detach().cpu().numpy().flatten()
        pred = pred.detach().cpu().numpy().flatten()

        fig_gt = plt.figure()
        ax = fig_gt.add_subplot(111, projection='3d')
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.scatter(x, y, gt, c='b', marker='o')
        ax.scatter(x, y, pred, c='r', marker='^')
        ax.set_title('Prediction vs Ground Truth')        

        buf = BytesIO()
        fig_gt.savefig(buf, format='png')
        buf.seek(0)
        fig_img = Image.open(buf)

        return {
            'plot': np.array(fig_img),
        }

    def log(self, step: int, split, eval_out: dict, writer: SummaryWriter):
        writer.add_image(f'{split}/plot', eval_out['plot'], step, dataformats='HWC')

    def test_ops(self, step: int, data: dict, out: dict, out_dir: Path) -> dict:
        pass