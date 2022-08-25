# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Various positional encodings for the transformer.
"""
import math
import torch
from torch import nn
import cv2
import numpy as np

from models.detr.util.misc import NestedTensor


class PositionEmbeddingSine_3D(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats_t = num_pos_feats/8*2
        self.num_pos_feats_s = num_pos_feats/8*3
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors

        mask = tensor_list.mask

        assert mask is not None
        #b, t, h, w = mask.shape
        #mask = mask.reshape(b, 1, h, w).repeat(1,4,1,1)
        not_mask = ~mask
        #print('mask1',mask[1, :,:,:])

        t_embed = not_mask.cumsum(1, dtype=torch.float32)
        y_embed = not_mask.cumsum(2, dtype=torch.float32)
        x_embed = not_mask.cumsum(3, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            t_embed = t_embed / (t_embed[:, -1:, :, :] + eps) * self.scale
            y_embed = y_embed / (y_embed[:, :, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, :, -1:] + eps) * self.scale


        t_dim_t = torch.arange(self.num_pos_feats_t, dtype=torch.float32, device=x.device)
        t_dim_t = self.temperature ** (2 * (t_dim_t // 2) / self.num_pos_feats_t)

        pos_t = t_embed[:, :, :, :, None] / t_dim_t

        s_dim_t = torch.arange(self.num_pos_feats_s, dtype=torch.float32, device=x.device)
        s_dim_t = self.temperature ** (2 * (s_dim_t // 2) / self.num_pos_feats_s)
        pos_x = x_embed[:, :, :, :, None] / s_dim_t
        pos_y = y_embed[:, :, :, :, None] / s_dim_t


        pos_x = torch.stack((pos_x[:, :, :, :, 0::2].sin(), pos_x[:, :, :, :, 1::2].cos()), dim=5).flatten(4)

        pos_y = torch.stack((pos_y[:, :, :, :, 0::2].sin(), pos_y[:, :, :, :, 1::2].cos()), dim=5).flatten(4)

        pos_t = torch.stack((pos_t[:, :, :, :, 0::2].sin(), pos_t[:, :, :, :, 1::2].cos()), dim=5).flatten(4)

        pos = torch.cat((pos_t, pos_y, pos_x), dim=4).permute(0, 4, 1, 2, 3)

        return pos

def build_position_encoding(hidden_dim):
    position_embedding = PositionEmbeddingSine_3D(hidden_dim, normalize=True)
    return position_embedding


