import time

import gin
import numpy as np
import torch
import torch.distributed as dist
import os

from pathlib import Path
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List
from itertools import islice

from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel

from .subtrainers import *

@gin.configurable()
class Trainer:

    def __init__(
        self,
        model,
        train_loader: DataLoader,
        eval_loader: DataLoader,
        test_loader: DataLoader = None,
        subtrainer : BaseSubtrainer = None,
        # configurable
        base_exp_dir: str = 'results',
        exp_name: str = 'DINOMVSFormer',
        effective_batch_size: int = 16,
        max_steps: int = 50000,
        log_step: int = 1000,
        eval_step: int = 1000,
        ckpt_step: int = 100,
        lr: float = 1e-4,
        resume: str = None,
        device="cuda:0",
        fp16=True,
        bf16=False,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.model = model.to(device)
        self.rank = rank
        self.is_master = self.rank == 0
        self.world_size = world_size
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.test_loader = test_loader

        self.max_steps = max_steps
        self.effective_batch_size = effective_batch_size
        # exp_dir
        self.exp_dir = Path(base_exp_dir) / exp_name
        self.val_dir = self.exp_dir / 'val'
        self.test_dir = self.exp_dir / 'test'
        self.ckpt_dir = self.exp_dir / 'ckpts'
        self.log_dir = self.exp_dir / 'logs'
        if self.is_master:
            self.val_dir.mkdir(parents=True, exist_ok=True)
            self.test_dir.mkdir(parents=True, exist_ok=True)
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)
            self.log_dir.mkdir(parents=True, exist_ok=True)
        self.is_dist = dist.is_initialized()
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        self.log_step = log_step
        self.eval_step = eval_step
        self.ckpt_step = ckpt_step
        self.device = device
        self.lr = lr
        self.fp16 = fp16
        self.bf16 = bf16

        self.subtrainer = subtrainer
        self.optimizer = self.subtrainer.get_optimizer(self.model, lr=self.lr)
        self.scheduler = self.subtrainer.get_scheduler(self.optimizer, max_steps=self.max_steps)
        # self.grad_scaler = torch.cuda.amp.GradScaler(2**10)
        # Save configure
        conf = gin.operative_config_str()
        if self.is_master:
            logger.info(conf)
            self.save_config(conf)
        
        if self.is_dist:
            self.model = DistributedDataParallel(self.model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

        self.start_step = 0
        if resume is not None:
            self.start_step = self.load_ckpt(resume)
                
        if self.is_dist:
            dist.barrier()

        if self.fp16:
            self.scaler = torch.cuda.amp.GradScaler()

    def train_iter(self, step: int, data: Dict, logging=False, optim_step=False):
        tic = time.time()

        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                data[k] = v.to(self.device)

        if self.fp16:
            with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16 if self.bf16 else torch.float16):
                out = self.model(data)
                loss_dict, invalid_data = self.subtrainer.compute_loss(data, out)
        else:
            out = self.model(data)
            loss_dict, invalid_data = self.subtrainer.compute_loss(data, out)

        # compute loss
        invalid_data = torch.tensor(int(invalid_data), device=self.device)
        dist.all_reduce(invalid_data, op=dist.ReduceOp.MAX)

        # if any process has invalid data, set loss in all processes to 0
        # because all process should calculate gradients on same tensors, should not diverge
        if invalid_data.any():
            logger.warning(f"Rank {self.rank} has invalid data at step {step}, setting loss to 0")
            for k in loss_dict:
                loss_dict[k] = torch.tensor(0., requires_grad=True, device=self.device)

        if self.fp16:
            self.scaler.scale(loss_dict['total_loss']).backward()
        else:
            loss_dict['total_loss'].backward()

        if optim_step:
            # update
            if self.fp16:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()

        # logging
        if logging:
            with torch.no_grad():
                iter_time = time.time() - tic
                remaining_time = (self.max_steps - step) * iter_time
                status = {
                    'lr': self.optimizer.param_groups[0]["lr"],
                    'step': step,
                    'iter_time': iter_time,
                    'ETA': remaining_time,
                }
                logger.info(status)
        return loss_dict

    def eval_iter(self, step: int, data: Dict, compute_loss=True):
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                data[k] = v.to(self.device)
        if self.fp16:
            with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16 if self.bf16 else torch.float16):
                out = self.model(data)
        else:
            out = self.model(data)

        # compute loss
        loss_dict = {}
        if compute_loss:
            loss_dict, _ = self.subtrainer.compute_loss(data, out)
        
        eval_out = self.subtrainer.eval_out(step, data, out)

        return loss_dict, eval_out


    def test_iter(self, data):
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                data[k] = v.to(self.device)
        if self.fp16:
            with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16 if self.bf16 else torch.float16):
                out = self.model(data)
        else:
            out = self.model(data)

        return out

    @torch.no_grad()
    def eval(self, step=0, log=False, compute_loss=True):
        logger.info(f"==> Rank : {self.rank}, Start evaluation on valset ...")
        self.model.eval()

        log_img_idxs = [i for i in range(len(self.eval_loader))][::max(1, len(self.eval_loader) // 6)]
        val_out = {}

        for val_step, data in enumerate(tqdm(self.eval_loader, total=len(self.eval_loader))):
            loss_dict, out = self.eval_iter(val_step, data, compute_loss)
            val_out_updated = self.subtrainer.eval_ops(val_step, data, val_out, loss_dict, out, self.val_dir)
            if val_out_updated is not None:
                val_out = val_out_updated

        self.subtrainer.post_eval_ops(
            self.is_dist, 
            val_out, 
            log, 
            self.writer, 
            is_master=self.is_master, 
            step=step)

    @torch.no_grad()
    def test(self, log=False):
        logger.info("==> Start evaluation on testset ...")
        self.model.eval()

        for test_step, data in enumerate(tqdm(self.test_loader, total=len(self.test_loader))):
            out = self.test_iter(data)
            self.subtrainer.test_ops(test_step, data, out, self.test_dir)

        # if log:
        #     logger.info(f"Mean error: {np.mean(error)}")
        #     self.writer.add_scalar('Test/mean_error', np.mean(error))

    def install_nan_checks(self, model):
        def mk(name):
            def hook(mod, inp, out):
                def bad(t): return torch.is_tensor(t) and (torch.isnan(t).any() or torch.isinf(t).any())
                ins  = inp if isinstance(inp, (tuple, list)) else (inp,)
                outs = out if isinstance(out, (tuple, list)) else (out,)
                if any(bad(t) for t in ins):  
                    breakpoint()
                    raise RuntimeError(f"NaN/Inf in INPUT of {name}")
                if any(bad(t) for t in outs): 
                    breakpoint()
                    raise RuntimeError(f"NaN/Inf in OUTPUT of {name}")

            return hook
        for n, m in model.named_modules():
            m.register_forward_hook(mk(n))

    def collect_fn(self, data):
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                data[k] = v.to(self.device)
        return data
    
    def fit(self):
        logger.info("==> Start training ...")
        self.install_nan_checks(self.model)

        # if self.fp16:
        #     with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16 if self.bf16 else torch.float16):
        #         torchsparse.tune(
        #             model=self.model,
        #             data_loader=self.train_loader,
        #             n_samples=10,
        #             collect_fn=self.collect_fn,
        #         )

        if hasattr(self.train_loader.sampler, 'set_epoch'):
            self.train_loader.sampler.set_epoch(0)

        # start iter_train_loader from start_step
        logger.info(f"Skipping {self.start_step} steps")
        iter_train_loader = iter(islice(self.train_loader, self.start_step, None))
        
        iters_per_batch = self.effective_batch_size // self.train_loader.batch_size
        logger.info(f"iters_per_batch: {iters_per_batch}")

        self.model.train()
        pbar = tqdm(initial=self.start_step, total=self.max_steps)
        for step in range(self.start_step, self.max_steps):
            avg_loss_dict = {}
            for mini_step in range(iters_per_batch):
                train_data = next(iter_train_loader)
                loss_dict = self.train_iter(
                    step,
                    data=train_data,
                    logging=(self.is_master and step % self.log_step == 0 and step > 0 and mini_step == 0),
                    optim_step=(mini_step == iters_per_batch - 1),
                )
                for k, v in loss_dict.items():
                    if k not in avg_loss_dict:
                        avg_loss_dict[k] = []
                    avg_loss_dict[k].append(v.item())

            avg_loss_dict = {k: np.mean(v) for k, v in avg_loss_dict.items()}
            for key in avg_loss_dict:
                temp_reduce = torch.tensor(avg_loss_dict[key]).to(self.device)
                dist.reduce(temp_reduce, 0, dist.ReduceOp.AVG)
                avg_loss_dict[key] = temp_reduce.item()
            if self.is_master:
                pbar.update(1)
                pbar.set_description(
                    f"rank: {self.rank}, loss: {avg_loss_dict['total_loss']:.4f}"
                )
                self.writer.add_scalars('Train/loss', avg_loss_dict, step)
                self.writer.add_scalar('Train/lr', self.optimizer.param_groups[0]["lr"], step)


            if step % self.log_step == 0:
                self.model.eval()
                _, out = self.eval_iter(step, train_data, compute_loss=False)
                if self.is_master:
                    self.subtrainer.log(step, "Train", out, self.writer)
                self.model.train()

            if self.is_master and step != 0 and step % self.ckpt_step == 0:
                self.save_ckpt(step)
                
            if step % self.eval_step == 0 and step != 0:
                dist.barrier()
                self.eval(step=step, log=True)
                self.model.train()

            dist.barrier()

        if self.is_master:
            logger.info('==> Training done!')
            self.save_ckpt(step)

    def save_config(self, config):
        dest = self.exp_dir / 'config.gin'
        # if dest.exists():
        #     return
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        with open(self.exp_dir / 'config.gin', 'w') as f:
            f.write(config)
        md_config_str = gin.config.markdown(config)
        self.writer.add_text("config", md_config_str)
        self.writer.flush()

    def save_ckpt(self, step):
        dest = self.ckpt_dir / ('model' + str(step).zfill(5) + '.ckpt')
        logger.info('==> Saving checkpoints to ' + str(dest))
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "step": step,
            },
            dest,
        )

    def load_ckpt(self, path):
        if not os.path.exists(path):
            logger.info(f"Checkpoint {path} does not exist!")
            exit(0)
        data = torch.load(path, map_location=self.device)
        dist_model = {}

        if self.is_dist:
            for k, v in data['model'].items():
                if k.startswith('module.'):
                    dist_model = data['model']
                    break
                dist_model['module.' + k] = v
        else:
            dist_model = data['model']
            torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(dist_model, 'module.')

        # self.model.load_state_dict(data['model'])
        self.model.load_state_dict(dist_model, strict=True)
        self.optimizer.load_state_dict(data['optimizer'])
        self.scheduler.load_state_dict(data['scheduler'])
        logger.info('==> Loading checkpoints from ' + str(path))

        return data['step'] + 1

