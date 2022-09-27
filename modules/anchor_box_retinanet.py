import torch
from math import sqrt as sqrt
from itertools import product as product
import numpy as np
from modules.utils import BufferList

class anchorBox(torch.nn.Module):
    """Compute anchorbox coordinates in center-offset form for each source
    feature map.
    """
    def __init__(self, sizes = [32, 64, 128, 256, 512],
                        ratios = np.asarray([0.5, 1 / 1., 2.0]),
                        strides = [8, 16, 32, 64, 128],
                        scales = np.array([1, 1.25992, 1.58740])):

        super(anchorBox, self).__init__()
        self.sizes = sizes
        self.ratios = ratios
        self.scales = scales
        self.strides = strides
        self.ar = len(self.ratios)*len(self.ratios)
        self.cell_anchors = BufferList(self._get_cell_anchors())
        
    def _get_cell_anchors(self):
        anchors = []
        for s1 in self.sizes:
            p_anchors = np.asarray(self._gen_generate_anchors_on_one_level(s1))
            p_anchors = torch.FloatTensor(p_anchors).cuda()
            anchors.append(p_anchors)

        return anchors
    
    # modified from https://github.com/fizyr/keras-retinanet/blob/master/keras_retinanet/utils/anchors.py
    # Copyright 2017-2018 Fizyr (https://fizyr.com)
    def _gen_generate_anchors_on_one_level(self, base_size=32):
        
        """
        Generate anchor (reference) windows by enumerating aspect ratios X
        scales w.r.t. a reference window.
        
        """

        num_anchors = len(self.ratios) * len(self.scales)

        # initialize output anchors
        anchors = np.zeros((num_anchors, 4))
        
        # print(self.scales)
        # scale base_size
        anchors[:, 2:] = base_size * np.tile(self.scales, (2, len(self.ratios))).T
        # print(anchors)
        # compute areas of anchors
        areas = anchors[:, 2] * anchors[:, 3]

        anchors[:, 2] = np.sqrt(areas / np.repeat(self.ratios, len(self.scales)))
        anchors[:, 3] = anchors[:, 2] * np.repeat(self.ratios, len(self.scales))
        # print(anchors)
        # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
        anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
        anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T
        # print(anchors)
        return anchors

    # forward from https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
    def forward(self, grid_sizes):
        
        anchors = []
        for size, stride, base_anchors in zip(grid_sizes, self.strides, self.cell_anchors):
            grid_height, grid_width = size
            device = base_anchors.device
            shifts_x = torch.arange(0, grid_width, dtype=torch.float32, device=device)
            shifts_y = torch.arange(0, grid_height, dtype=torch.float32, device=device)
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x) 
            shift_x = (shift_x.reshape(-1) + 0.5) * stride
            shift_y = (shift_y.reshape(-1) + 0.5) * stride
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)
            anchors.append( (shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4) )

        return torch.cat(anchors, 0)
        
