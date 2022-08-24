# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable
import time
import torch

import numpy as np
from utils.utils import AverageMeter
from evaluates.evaluate_ava import STDetectionEvaluater


@torch.no_grad()
def evaluate(cfg, model: torch.nn.Module, criterion: torch.nn.Module, postprocessors,
             data_loader: Iterable, ddp_params: dict, epoch: int, writer=None):
    #
    batch_time = AverageMeter()
    data_time = AverageMeter()
    class_err = AverageMeter()
    losses_box = AverageMeter()
    losses_giou = AverageMeter()
    losses_ce = AverageMeter()
    losses_avg = AverageMeter()
    losses_ce_b = AverageMeter()

    end = time.time()
    model.eval()
    criterion.eval()

    # metric_logger = utils.MetricLogger(delimiter="  ")
    # metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    buff_output = []
    buff_anno = []
    buff_id = []
    buff_GT_label = []
    buff_GT_anno = []
    buff_GT_id = []

    for idx, data in enumerate(data_loader):
        data_time.update(time.time() - end)

        # for samples, targets in metric_logger.log_every(data_loader, print_freq, epoch, ddp_params, writer, header):
        device = "cuda:" + str(ddp_params["gpu"])
        samples = data[0]
        targets = data[1]

        samples = samples.to(device)

        batch_id = [t["image_id"] for t in targets]

        for t in targets:
            del t["image_id"]

        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)

        loss_dict = criterion(outputs, targets)

        weight_dict = criterion.weight_dict

        orig_target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        scores, boxes = postprocessors['bbox'](outputs, orig_target_sizes)
        for bidx in range(scores.shape[0]):
            frame_id = batch_id[bidx][0]
            key_pos = batch_id[bidx][1]

            buff_output.append(scores[bidx, key_pos * cfg.CONFIG.MODEL.QUERY_NUM:(key_pos + 1) * cfg.CONFIG.MODEL.QUERY_NUM, :])
            buff_anno.append(boxes[bidx, key_pos * cfg.CONFIG.MODEL.QUERY_NUM:(key_pos + 1) * cfg.CONFIG.MODEL.QUERY_NUM, :])

            for l in range(cfg.CONFIG.MODEL.QUERY_NUM):
                buff_id.extend([frame_id])

            raw_idx = (targets[bidx]["raw_boxes"][:, 1] == key_pos).nonzero().squeeze()

            val_label = targets[bidx]["labels"][raw_idx]
            val_label = val_label.reshape(-1, val_label.shape[-1])
            raw_boxes = targets[bidx]["raw_boxes"][raw_idx]
            raw_boxes = raw_boxes.reshape(-1, raw_boxes.shape[-1])
            # print('raw_boxes',raw_boxes.shape)

            buff_GT_label.append(val_label.detach().cpu().numpy())
            buff_GT_anno.append(raw_boxes.detach().cpu().numpy())

            img_id_item = [batch_id[int(raw_boxes[x, 0] - targets[0]["raw_boxes"][0, 0])][0] for x in
                           range(len(raw_boxes))]  # JJ: Why?

            buff_GT_id.extend(img_id_item)

        batch_time.update(time.time() - end)
        end = time.time()

        if (ddp_params['rank'] == 0):
            print_string = 'Epoch: [{0}][{1}/{2}]'.format(epoch, idx + 1, len(data_loader))
            print(print_string)
            print_string = 'data_time: {data_time:.3f}, batch time: {batch_time:.3f}'.format(
                data_time=data_time.val,
                batch_time=batch_time.val)
            print(print_string)

            # reduce losses over all GPUs for logging purposes
            # loss_dict_reduced = utils.reduce_dict(loss_dict)

            # reduce on single GPU
            loss_dict_reduced = loss_dict
            loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                          for k, v in loss_dict_reduced.items()}
            loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                        for k, v in loss_dict_reduced.items() if k in weight_dict}
            losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

            loss_value = losses_reduced_scaled.item()

            losses_avg.update(loss_value, len(targets))
            losses_box.update(loss_dict_reduced['loss_bbox'].item(), len(targets))
            losses_giou.update(loss_dict_reduced['loss_giou'].item(), len(targets))
            losses_ce.update(loss_dict_reduced['loss_ce'].item(), len(targets))
            class_err.update(loss_dict_reduced['class_error'], len(targets))

            if cfg.CONFIG.MATCHER.BNY_LOSS:
                losses_ce_b.update(loss_dict_reduced['loss_ce_b'].item(), len(targets))

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping eval".format(loss_value))
                print(loss_dict_reduced)
                sys.exit(1)
            print_string = 'class_error: {class_error:.3f}, loss: {loss:.3f}, loss_bbox: {loss_bbox:.3f}, loss_giou: {loss_giou:.3f}, loss_ce: {loss_ce:.3f}, loss_ce_b: {loss_ce_b:.3f}'.format(
                class_error=class_err.avg,
                loss=losses_avg.avg,
                loss_bbox=losses_box.avg,
                loss_giou=losses_giou.avg,
                loss_ce=losses_ce.avg,
                loss_ce_b=losses_ce_b.avg,
                # cardinality_error=loss_dict_reduced['cardinality_error']
            )
            print(print_string)

    if cfg.DDP_CONFIG.GPU_WORLD_RANK == 0:
        writer.add_scalar('val/class_error', class_err.avg, epoch)
        writer.add_scalar('val/totall_loss', losses_avg.avg, epoch)
        writer.add_scalar('val/loss_bbox', losses_box.avg, epoch)
        writer.add_scalar('val/loss_giou', losses_giou.avg, epoch)
        writer.add_scalar('val/loss_ce', losses_ce.avg, epoch)
        writer.add_scalar('val/loss_ce_b', losses_ce_b.avg, epoch)


    buff_output = np.concatenate(buff_output, axis=0)
    buff_anno = np.concatenate(buff_anno, axis=0)
    buff_GT_label = np.concatenate(buff_GT_label, axis=0)
    buff_GT_anno = np.concatenate(buff_GT_anno, axis=0)
    print(buff_output.shape, buff_anno.shape, len(buff_id), buff_GT_anno.shape, buff_GT_label.shape, len(buff_GT_id))

    tmp_path = '{}/{}/{}.txt'
    with open(tmp_path.format(cfg.CONFIG.LOG.BASE_PATH, cfg.CONFIG.LOG.RES_DIR, cfg.DDP_CONFIG.GPU_WORLD_RANK), 'w') as f:
        for x in range(len(buff_id)):
            data = np.concatenate([buff_anno[x], buff_output[x]])
            f.write("{} {}\n".format(buff_id[x], data.tolist()))
    tmp_GT_path = '{}/{}/GT_{}.txt'
    with open(tmp_GT_path.format(cfg.CONFIG.LOG.BASE_PATH, cfg.CONFIG.LOG.RES_DIR, cfg.DDP_CONFIG.GPU_WORLD_RANK), 'w') as f:
        for x in range(len(buff_GT_id)):
            data = np.concatenate([buff_GT_anno[x], buff_GT_label[x]])
            f.write("{} {}\n".format(buff_GT_id[x], data.tolist()))

    # write files and align all workers
    torch.distributed.barrier()
    # aggregate files
    Map_ = 0
    # aggregate files
    if cfg.DDP_CONFIG.GPU_WORLD_RANK == 0:
        # read results
        evaluater = STDetectionEvaluater("/xxx/ava_action_list_v2.1_for_activitynet_2018.pbtxt.txt")
        file_path_lst = [tmp_GT_path.format(cfg.CONFIG.LOG.BASE_PATH, cfg.CONFIG.LOG.RES_DIR, x) for x in range(ddp_params['world_size'])]
        evaluater.load_GT_from_path(file_path_lst)
        file_path_lst = [tmp_path.format(cfg.CONFIG.LOG.BASE_PATH, cfg.CONFIG.LOG.RES_DIR, x) for x in range(ddp_params['world_size'])]
        evaluater.load_detection_from_path(file_path_lst)
        mAP, metrics = evaluater.evaluate()
        print(metrics)
        print_string = 'mAP: {mAP:.5f}'.format(mAP=mAP[0])
        print(print_string)
        print(mAP)
        writer.add_scalar('val/val_mAP_epoch', mAP[0], epoch)
        Map_ = mAP[0]
    torch.distributed.barrier()
    return Map_