import argparse
import gin
import os
import torch
import torch.distributed as dist
import numpy as np
import random

from datetime import datetime
import datetime

from glob import glob
from loguru import logger
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler

from datasets import *
from trainer.trainer import Trainer
from models import *
from trainer.subtrainers import *

torch.manual_seed(4)
np.random.seed(4)
random.seed(4)

@gin.configurable
def main(
    seed: int = 4,
    rank=0,
    world_size=1,
    num_workers: int = 0,
    batch_size: int = 4,
    helper_type : BaseSubtrainer = None,
    model_type = None,
    dataset_type = [PolynomialDataset],
):
    device = torch.device(f"cuda:{rank}")
    logger.info("==> Init dataloader ...")
    # path = "/nas/shreyansh/DataMatterport3D/Generated/"
    # train_dataset = APDDataset(path, split='train')

    train_datasets = []
    for dataset_cls in dataset_type:
        train_datasets.append(dataset_cls(split='train'))

    train_dataset = ConcatDataset(train_datasets)
    sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=(sampler is None),
        sampler=sampler,
        pin_memory=True,
        worker_init_fn=None,
        pin_memory_device='cuda',
        prefetch_factor=2,
    )

    val_dataset = dataset_type[0](split='val')
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
        worker_init_fn=None,
        pin_memory_device='cuda',
        prefetch_factor=2,
    )

    test_loader = None

    logger.info("==> Init model ...")
    model = model_type(device=device)
    helper = helper_type(device)
    trainer = Trainer(model, train_loader, val_loader, test_loader, subtrainer=helper, device=device, rank=rank, world_size=world_size)
    # if rank == 0:
    #     trainer.eval(step=18000, save_pcld=True, log=True)
    trainer.fit()
    if rank == 0:
        trainer.eval(step=trainer.max_steps, log=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ginc",
        action="append",
        help="gin config file",
    )
    args = parser.parse_args()

    ginbs = []
    gin.parse_config_files_and_bindings(args.ginc,
                                        ginbs,
                                        finalize_config=False)

    exp_name = gin.query_parameter("Trainer.exp_name")

    gin.bind_parameter("Trainer.exp_name", exp_name)
    gin.finalize()

    global_rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    print(f"Local Rank : {local_rank}, Global Rank : {global_rank}, World Size : {world_size}")
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(minutes = 15))

    torch.multiprocessing.set_start_method('forkserver')
    main(rank=local_rank, world_size=world_size)
