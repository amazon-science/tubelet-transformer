import argparse
import datetime
import time

import torch
import torch.optim
from tensorboardX import SummaryWriter

from models.tuber_ava import build_model
from utils.model_utils import deploy_model, load_model, save_checkpoint
from utils.video_action_recognition import train_tuber_detection, validate_tuber_detection
from pipelines.video_action_recognition_config import get_cfg_defaults
from pipelines.launch import spawn_workers
from utils.utils import build_log_dir
from datasets.ava_frame import build_dataloader
from utils.lr_scheduler import build_scheduler


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
    for epoch in range(cfg.CONFIG.TRAIN.START_EPOCH, cfg.CONFIG.TRAIN.EPOCH_NUM):
        if cfg.DDP_CONFIG.DISTRIBUTED:
            train_sampler.set_epoch(epoch)

        train_tuber_detection(cfg, model, criterion, train_loader, optimizer, epoch, cfg.CONFIG.LOSS_COFS.CLIPS_MAX_NORM, lr_scheduler, writer)

        if cfg.DDP_CONFIG.GPU_WORLD_RANK == 0 and (
                epoch % cfg.CONFIG.LOG.SAVE_FREQ == 0 or epoch == cfg.CONFIG.TRAIN.EPOCH_NUM - 1):
            save_checkpoint(cfg, epoch, model, max_accuracy, optimizer, lr_scheduler)

        if epoch % cfg.CONFIG.VAL.FREQ == 0 or epoch == cfg.CONFIG.TRAIN.EPOCH_NUM - 1:
            validate_tuber_detection(cfg, model, criterion, postprocessors, val_loader, epoch, writer)

    if writer is not None:
        writer.close()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train video action recognition transformer models.')
    parser.add_argument('--config-file',
                        default='/xxx/TubeR_AVA_v2.1_CSN-152.yaml',
                        help='path to config file.')
    args = parser.parse_args()

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)
    spawn_workers(main_worker, cfg)
