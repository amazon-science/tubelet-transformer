import argparse
import datetime
import time

import torch
import torch.optim
from tensorboardX import SummaryWriter

from models.tuber_ava import build_model
from utils.model_utils import deploy_model, load_model, save_checkpoint
from utils.video_action_recognition import validate_tuber_detection
from pipelines.video_action_recognition_config import get_cfg_defaults
from pipelines.launch import spawn_workers
from utils.utils import build_log_dir
from datasets.ava_frame import build_dataloader


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
    _, test_loader, _, test_sampler,_ = build_dataloader(cfg)

    # docs: add resume option
    if not cfg.CONFIG.MODEL.LOAD: raise ("model dir not found")
    model, _ = load_model(model, cfg, load_fc=cfg.CONFIG.MODEL.LOAD_FC)

    print('Start training...')
    start_time = time.time()
    validate_tuber_detection(cfg, model, criterion, postprocessors, test_loader, 0, writer)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('testing time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train video action recognition transformer models.')
    parser.add_argument('--config-file',
                        default='/xxx/TubeR_AVA_v2.2_CSN-152.yaml',
                        help='path to config file.')
    args = parser.parse_args()

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)
    spawn_workers(main_worker, cfg)
