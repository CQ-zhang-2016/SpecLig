#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import argparse

import yaml
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import functools
print = functools.partial(print, flush=True)

from utils.logger import print_log
from utils.random_seed import setup_seed, SEED
from utils.config_utils import overwrite_values
from utils import register as R

########### Import your packages below ##########
from trainer import create_trainer
from data import create_dataset, create_dataloader
from utils.nn_utils import count_parameters
import models
import torch.multiprocessing as mp


def parse():
    parser = argparse.ArgumentParser(description='training')

    # device
    parser.add_argument('--gpus', type=int, required=True, help='gpu to use, -1 for cpu')

    # config
    parser.add_argument('--config', type=str, required=True, help='Path to the yaml configure')
    parser.add_argument('--seed', type=int, default=SEED, help='Random seed')

    return parser.parse_known_args()


def main(rank, args, opt_args):

    # load config
    config = yaml.safe_load(open(args.config, 'r'))
    config = overwrite_values(config, opt_args)

    ########## define your model #########
    model = R.construct(config['model'])
    if len(config.get('load_ckpt', '')):
        model.load_state_dict(torch.load(config['load_ckpt'], map_location='cpu').state_dict())
        print_log(f'Loaded weights from {config["load_ckpt"]}')

    ########### load your train / valid set ###########
    train_set, valid_set, _ = create_dataset(config['dataset'])

    ########## define your trainer/trainconfig #########
    if args.gpus > 1:
        device = torch.device(f'cuda:{rank}')
        torch.cuda.set_device(rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=args.gpus, rank=rank) # set by torchrun

    if rank <= 0:
        print_log(f'Number of parameters: {count_parameters(model) / 1e6} M')
    
    train_loader = create_dataloader(train_set, config['dataloader'].get('train', config['dataloader']), args.gpus, rank) # <-- 添加这行
    valid_loader = create_dataloader(valid_set, config['dataloader'].get('valid', config['dataloader']))
    
    trainer = create_trainer(config, model, train_loader, valid_loader)
    trainer.train(args.gpus, rank)

    torch.distributed.destroy_process_group()

if __name__ == '__main__':
    args, opt_args = parse()
    print_log(f'Overwritting args: {opt_args}')
    setup_seed(args.seed)
    # torch.autograd.set_detect_anomaly(True)
    # 配置分布式通信环境变量
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'

    world_size = args.gpus if args.gpus > 0 else 1
    if args.gpus > 1:
        mp.spawn(
            main,
            args=(args, opt_args),
            nprocs=world_size,
            join=True
        )
    else:
        main(-1, args, opt_args)
