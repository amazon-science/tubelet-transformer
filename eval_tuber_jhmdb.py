import argparse
import datetime
import time

import torch
import torch.optim
from tensorboardX import SummaryWriter

from models.tuber_ava import build_model
from utils.model_utils import deploy_model, load_model
from utils.video_action_recognition import validate_tuber_ucf_detection
from pipelines.video_action_recognition_config import get_cfg_defaults
from pipelines.launch import spawn_workers
from utils.utils import build_log_dir
from datasets.jhmdb_frame import build_dataloader
from utils.lr_scheduler import build_scheduler
import os


def main_worker(cfg):
    # create tensorboard and logs
    if cfg.DDP_CONFIG.GPU_WORLD_RANK == 0:
        tb_logdir = build_log_dir(cfg)
        writer = SummaryWriter(log_dir=tb_logdir)
    else:
        writer = None
    # cfg.freeze()

    # create model
    print('Creating TubeR model: %s' % cfg.CONFIG.MODEL.NAME)
    model, criterion, postprocessors = build_model(cfg)
    model = deploy_model(model, cfg, is_tuber=True)
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of parameters in the model: %6.2fM' % (num_parameters / 1000000))

    # create dataset and dataloader
    train_loader, val_loader, train_sampler, val_sampler, mg_sampler = build_dataloader(cfg)

    print("test sampler", len(train_loader))
    # create criterion
    criterion = criterion.cuda()

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and "class_embed" not in n and "query_embed" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": cfg.CONFIG.TRAIN.LR_BACKBONE,
        },
        {
            "params": [p for n, p in model.named_parameters() if "class_embed" in n and p.requires_grad],
            "lr": cfg.CONFIG.TRAIN.LR, #10
        },
        {
            "params": [p for n, p in model.named_parameters() if "query_embed" in n and p.requires_grad],
            "lr": cfg.CONFIG.TRAIN.LR, #10
        },
    ]

    # param_dicts = model.parameters()

    # create optimizer
    optimizer = torch.optim.AdamW(param_dicts, lr=cfg.CONFIG.TRAIN.LR, weight_decay=cfg.CONFIG.TRAIN.W_DECAY)

    # create lr scheduler
    if cfg.CONFIG.TRAIN.LR_POLICY == "step":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,60], gamma=0.1)
    else:
        lr_scheduler = build_scheduler(cfg, optimizer, len(train_loader))

    # docs: add resume option
    if cfg.CONFIG.MODEL.LOAD:
        model, _ = load_model(model, cfg, load_fc=cfg.CONFIG.MODEL.LOAD_FC)

    print('Start training...')
    start_time = time.time()
    max_accuracy = 0.0
    validate_tuber_ucf_detection(cfg, model, criterion, postprocessors, val_loader, 0, writer)
    print("eval finished")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train video action recognition transformer models.')
    parser.add_argument('--config-file',
                        default='/xxx/Tuber_JHMDB_CSN-152.yaml',
                        help='path to config file.')

    args = parser.parse_args()

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)
    spawn_workers(main_worker, cfg)
