import torch
from math import sqrt as sqrt
from itertools import product as product
import numpy as np
from modules.utils import BufferList


class anchorBox(torch.nn.Module):
    """Compute anchorbox coordinates in center-offset form for each source
    feature map.
    """
    def __init__(self, aspect_ratios =[0.5, 1 / 1., 1.5],
                    scale_ratios = [1.,]):

        super(anchorBox, self).__init__()
        self.aspect_ratios = aspect_ratios
        self.scale_ratios = scale_ratios
        self.default_sizes= [0.01, 0.06, 0.2, 0.4, 0.85]
        self.anchor_boxes = len(self.aspect_ratios)*len(self.scale_ratios)
        self.ar = self.anchor_boxes
        self.num_anchors = self.ar
        self.cell_anchors = BufferList(self._get_cell_anchors())
        
    def _get_cell_anchors(self):
        anchors = []
        base_anchors = np.asarray([[0.0000, 0.0000, 0.0141, 0.0365],
                                    [0.0000, 0.0000, 0.0178, 0.0614],
                                    [0.0000, 0.0000, 0.0343, 0.0487],
                                    [0.0000, 0.0000, 0.0450, 0.1475],
                                    [0.0000, 0.0000, 0.0284, 0.0986],
                                    [0.0000, 0.0000, 0.0667, 0.0691],
                                    [0.0000, 0.0000, 0.0699, 0.2465],
                                    [0.0000, 0.0000, 0.1629, 0.1744],
                                    [0.0000, 0.0000, 0.1110, 0.1124],
                                    [0.0000, 0.0000, 0.1349, 0.3740],
                                    [0.0000, 0.0000, 0.2773, 0.3713],
                                    [0.0000, 0.0000, 0.2406, 0.2320],
                                    [0.0000, 0.0000, 0.3307, 0.6395],
                                    [0.0000, 0.0000, 0.7772, 0.6261],
                                    [0.0000, 0.0000, 0.4732, 0.3153]])
        
        for s1 in range(len(self.default_sizes)):
            p_anchors = base_anchors[s1*3:(s1+1)*3,:]
            p_anchors[:,:2] = p_anchors[:,:2]-p_anchors[:,2:]/2.0
            p_anchors[:,2:] = p_anchors[:,2:]/2.0
            p_anchors = torch.FloatTensor(p_anchors).cuda()
            # print(p_anchors)
            anchors.append(p_anchors)

        return anchors
    
    # based on forward from https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
    def forward(self, grid_sizes):
        
        anchors = []
        for size, base_anchors in zip(grid_sizes, self.cell_anchors):
            grid_height, grid_width = size
            stride_h = 1.0/grid_height
            stride_w = 1.0/grid_width
            device = base_anchors.device
            shifts_x = torch.arange(0, grid_width, dtype=torch.float32, device=device).cuda()
            shifts_y = torch.arange(0, grid_height, dtype=torch.float32, device=device).cuda() 
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x) 
            shift_x = (shift_x.reshape(-1) + 0.5) * stride_w
            shift_y = (shift_y.reshape(-1) + 0.5) * stride_h
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)
            anchors.append( (shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4) )
        
        anchors = torch.cat(anchors, 0)
        anchors.clamp_(max=1, min=0)
        return anchors
        
