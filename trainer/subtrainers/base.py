from torch.utils.tensorboard import SummaryWriter
import torch
from utils.utils import CosineWarmupScheduler
from loguru import logger
import gin
from pathlib import Path
from typing import Tuple
import torch.distributed as dist
import numpy as np

@gin.configurable()
class BaseSubtrainer:
    def __init__(self, device):
        self.device = device

    def compute_loss(self, data: dict, out: dict) -> Tuple[dict, bool]:
        pass
    
    def eval_out(self, step: int, data: dict, out: dict) -> dict:
        pass

    def log(self, step: int, split, eval_out: dict, writer: SummaryWriter):
        pass

    def test_ops(self, step: int, data: dict, out: dict, out_dir: Path) -> dict:
        pass
    
    def get_optimizer(self, model, lr):
        param_groups = []
        param_groups.append({"params": [p for n, p in model.named_parameters()], "lr": lr})
        optimizer = torch.optim.Adam(param_groups)
        return optimizer

    @gin.configurable()
    def get_scheduler(self, optimizer, warmup_iters:int = 200, max_steps:int = 10000):
        scheduler = CosineWarmupScheduler(
            optimizer,
            warmup_iters=warmup_iters,
            max_iters=max_steps,
        )
        logger.info(f"==> Using CosineWarmupScheduler with warmup_iters={warmup_iters}, max_iters={max_steps}")
        return scheduler
    
    def eval_ops(self, step: int, data: dict, val_out: dict, loss_dict: dict, out:dict, outdir: Path):
        if val_out is None:
            val_out = {}
            
        if 'avg_loss_dict' not in val_out:
            val_out['avg_loss_dict'] = {}
        for k, v in loss_dict.items():
            if k not in val_out['avg_loss_dict']:
                val_out['avg_loss_dict'][k] = []
            val_out['avg_loss_dict'][k].append(v.item())

        return val_out

    def post_eval_ops(self, is_dist: bool, val_out: dict, log: bool, writer: SummaryWriter, is_master: bool, step: int):
        avg_loss_dict = val_out['avg_loss_dict']


        avg_loss_dict = {k: np.mean(v) for k, v in avg_loss_dict.items()}
        for key in avg_loss_dict:
            temp_reduce = torch.tensor(avg_loss_dict[key], device=self.device)
            dist.reduce(temp_reduce, 0, dist.ReduceOp.AVG)
            avg_loss_dict[key] = temp_reduce.item()
            
        if is_master:
            if log:
                writer.add_scalars(f'Val/loss', avg_loss_dict, step)

            # logger.info(avg_loss_dict)
            # logger.info("==> Evaluation done!")

        pass