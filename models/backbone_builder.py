# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
import sys
import numpy as np

from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from models.transformer.util.misc import NestedTensor, is_main_process

from models.transformer.position_encoding import build_position_encoding

from models.backbones.ir_CSN_50 import build_CSN
from models.backbones.ir_CSN_152 import build_CSN as build_CSN_152
from models.transformer.transformer_layers import LSTRTransformerDecoder, LSTRTransformerDecoderLayer, layer_norm


class Backbone(nn.Module):

    def __init__(self, train_backbone: bool, num_channels: int, position_embedding, cfg):
        super().__init__()

        if cfg.CONFIG.MODEL.BACKBONE_NAME == 'CSN-152':
            print("CSN-152 backbone")
            self.body = build_CSN_152(cfg)
        else:
            print("CSN-50 backbone")
            self.body = build_CSN(cfg)
        self.position_embedding = position_embedding
        for name, parameter in self.body.named_parameters():
            if not train_backbone:
                parameter.requires_grad_(False)
        self.ds = cfg.CONFIG.MODEL.SINGLE_FRAME
        if cfg.CONFIG.MODEL.SINGLE_FRAME:
            if cfg.CONFIG.MODEL.TEMPORAL_DS_STRATEGY == 'avg':
                self.pool = nn.AvgPool3d((cfg.CONFIG.MODEL.TEMP_LEN // cfg.CONFIG.MODEL.DS_RATE, 1, 1))
                # print("avg pool: {}".format(cfg.CONFIG.MODEL.TEMP_LEN // cfg.CONFIG.MODEL.DS_RATE))
            elif cfg.CONFIG.MODEL.TEMPORAL_DS_STRATEGY == 'max':
                self.pool = nn.MaxPool3d((cfg.CONFIG.MODEL.TEMP_LEN // cfg.CONFIG.MODEL.DS_RATE, 1, 1))
                print("max pool: {}".format(cfg.CONFIG.MODEL.TEMP_LEN // cfg.CONFIG.MODEL.DS_RATE))
            elif cfg.CONFIG.MODEL.TEMPORAL_DS_STRATEGY == 'decode':
                self.query_pool = nn.Embedding(1, 2048)
                self.pool_decoder = LSTRTransformerDecoder(
                    LSTRTransformerDecoderLayer(d_model=2048, nhead=8, dim_feedforward=2048, dropout=0.1), 1,
                    norm=layer_norm(d_model=2048, condition=True))

        self.num_channels = num_channels
        self.backbone_name = cfg.CONFIG.MODEL.BACKBONE_NAME
        self.temporal_ds_strategy = cfg.CONFIG.MODEL.TEMPORAL_DS_STRATEGY

    def forward(self, tensor_list: NestedTensor):
        if "SlowFast" in self.backbone_name:
            xs, xt = self.body([tensor_list.tensors[:, :, ::4, ...], tensor_list.tensors])
            xs_orig = xt
        elif "TPN" in self.backbone_name:
            xs, xt = self.body(tensor_list.tensors)
            xs_orig = xt
        else:
            xs, xt = self.body(tensor_list.tensors)
            xs_orig = xs
        # if self.ds: xs = self.avg_pool(xs)
        bs, ch, t, w, h = xs.shape
        if self.ds:
            if self.temporal_ds_strategy == 'avg' or self.temporal_ds_strategy == 'max':
                xs = self.pool(xs)
            elif self.temporal_ds_strategy == 'decode':
                xs = xs.view(bs, ch, t, w * h).permute(2, 0, 3, 1).contiguous().view(t, bs * w * h, ch)
                query_embed = self.query_pool.weight.unsqueeze(1).repeat(1, bs * w * h, 1)
                xs = self.pool_decoder(query_embed, xs)
                xs = xs.view(1, bs, w * h, ch).permute(1, 3, 0, 2).contiguous().view(bs, ch, 1, w, h)
            else:
                xs = xs[:, :, t // 2: t // 2 + 1, ...]
        out: Dict[str, NestedTensor] = {}
        m = tensor_list.mask
        assert m is not None

        mask = F.interpolate(m[None].float(), size=xs.shape[-2:]).to(torch.bool)[0]
        mask = mask.unsqueeze(1).repeat(1,xs.shape[2],1,1)

        out = [NestedTensor(xs, mask)]
        pos = [self.position_embedding(NestedTensor(xs, mask))]
        return out, pos, xs_orig


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs, xl = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():

            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))
        return out, pos, xl


def build_backbone(cfg):
    position_embedding = build_position_encoding(cfg.CONFIG.MODEL.D_MODEL)
    model = Backbone(cfg.CONFIG.TRAIN.LR_BACKBONE > 0, cfg.CONFIG.MODEL.DIM_FEEDFORWARD, position_embedding, cfg)
    # model = Joiner(backbone, position_embedding)
    return model
