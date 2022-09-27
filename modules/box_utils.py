import torch, pdb, math
import numpy as np
import torchvision


def match_anchors_wIgnore(gt_boxes, gt_labels, anchors, pos_th=0.5, nge_th=0.4, variances=[0.1, 0.2], seq_len=1):
    # pdb.set_trace()
    # pdb.set_trace()
    num_mt = int(gt_labels.size(0)/seq_len)
    
    # pdb.set_trace()
    seq_overlaps =[]
    inds = torch.LongTensor([m*seq_len for m in range(num_mt)])  
    # print('indexs device', inds.device)
    # print(inds, num_mt)
    ## get indexes of first frame in seq for each microtube
    gt_labels = gt_labels[inds]
    # print('gtb', gt_boxes)
    # print('anchors', anchors[:10])
    
    for s in range(seq_len):
        seq_overlaps.append(jaccard(gt_boxes[inds+s, :], anchors))
    # pdb.set_trace()
    overlaps = seq_overlaps[0]
    # print('overlap max ', overlaps.max())
    ## Compute average overlap
    for s in range(seq_len-1):
        overlaps = overlaps + seq_overlaps[s+1]
    overlaps = overlaps/float(seq_len)
    # pdb.set_trace()
    best_anchor_overlap, best_anchor_idx = overlaps.max(1, keepdim=True)
    
    # print('MIN VAL::', best_anchor_overlap.min().item())
    # if best_anchor_overlap.min().item()<0.25:
    #     print('MIN VAL::', best_anchor_overlap.min().item())
    #     print('lower than o.5', best_anchor_overlap, gt_boxes)
    # [1,num_anchors] best ground truth for each anchor
    
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_anchor_idx.squeeze_(1)
    best_anchor_overlap.squeeze_(1)
    best_truth_overlap.index_fill_(0, best_anchor_idx, 2)  # ensure best anchor
    # ensure every gt matches with its anchor of max overlap
    for j in range(best_anchor_idx.size(0)):
        best_truth_idx[best_anchor_idx[j]] = j

    conf = gt_labels[best_truth_idx] + 1  # assigned nearest class label
    conf[best_truth_overlap < pos_th] = -1  # label as ignore
    conf[best_truth_overlap < nge_th] = 0  # label as background

    for s in range(seq_len):
        st = gt_boxes[inds + s, :]
        matches = st[best_truth_idx]  # Shape: [num_anchors,4]
        if s == 0:
            loc = encode(matches, anchors[:, s * 4:(s + 1) * 4], variances)  
            # Shape: [num_anchors, 4] -- encode the gt boxes for frame i
        else:
            temp = encode(matches, anchors[:, s * 4:(s + 1) * 4], variances)
            loc = torch.cat([loc, temp], 1)  # shape: [num_anchors x 4 * seql_len] : stacking the location targets for different frames
    # pdb.set_trace()
    return conf, loc
            

def hard_negative_mining(loss, labels, neg_pos_ratio):
    """
    It used to suppress the presence of a large number of negative prediction.
    It works on image level not batch level.
    For any example/image, it keeps all the positive predictions and
     cut the number of negative predictions to make sure the ratio
     between the negative examples and positive examples is no more
     the given ratio for an image.
    Args:
        loss (N, num_anchors): the loss for each example.
        labels (N, num_anchors): the labels.
        neg_pos_ratio:  the ratio between the negative examples and positive examples.
    
    """
    
    pos_mask = labels > 0
    num_pos = pos_mask.long().sum(dim=1, keepdim=True)
    num_neg = num_pos * neg_pos_ratio

    loss[pos_mask] = -math.inf
    _, indexes = loss.sort(dim=1, descending=True)
    _, orders = indexes.sort(dim=1)
    neg_mask = orders < num_neg
    return pos_mask | neg_mask
    

def point_form(boxes):
    """ Convert anchor_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.
    Args:
        boxes: (tensor) center-size default boxes from anchorbox layers.
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, :2] - boxes[:, 2:]/2,     # xmin, ymin
                     boxes[:, :2] + boxes[:, 2:]/2), 1)  # xmax, ymax


def center_size(boxes):
    """ Convert anchor_boxes to (cx, cy, w, h)
    representation for comparison to center-size form ground truth data.
    Args:
        boxes: (tensor) point_form boxes
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, 2:] + boxes[:, :2])/2,  # cx, cy
                     boxes[:, 2:] - boxes[:, :2], 1)  # w, h


def intersect(box_a, box_b):
    """ 
    
    We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    
    """
    # print(box_a, box_b)
    A = box_a.size(0)
    B = box_b.size(0)
    # pdb.set_trace()
    # print(box_a.type(), box_b.type())
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) anchor boxes from anchorbox layers, Shape: [num_anchors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    # pdb.set_trace()
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    
    union = area_a + area_b - inter
    min_union = union.min()

    # print('minnin ', min_union, union)

    return inter / union  # [A,B]


def get_ovlp_cellwise(overlaps):
    feature_maps = [38, 19, 10, 5, 3, 1]
    aratios = [4, 6, 6, 6, 4, 4]
    dim = 0
    for f in feature_maps:
        dim += f*f
    out_ovlp = np.zeros(dim)
    count = 0
    st = 0
    for k, f in enumerate(feature_maps):
        ar = aratios[k]
        for i in range(f*f):
            et = st+ar
            ovlps_tmp = overlaps[0, st:et]
            #pdb.set_trace()
            out_ovlp[count] = max(ovlps_tmp)
            count += 1
            st = et
    assert count == dim

    return out_ovlp


def encode(matched, anchors, variances):
    
    """
    
    Encode the variances from the anchorbox layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the anchor boxes.
    Args:
        matched: (tensor) Coords of ground truth for each anchor in point-form
            Shape: [num_anchors, 4].
        anchors: (tensor) anchor boxes in center-offset form
            Shape: [num_anchors,4].
        variances: (list[float]) Variances of anchorboxes
    
    Return:
        encoded boxes (tensor), Shape: [num_anchors, 4]
    
    """
    
    TO_REMOVE = 1 if anchors[0,2]>1 else 0 # TODO remove
    ex_widths = anchors[:, 2] - anchors[:, 0] + TO_REMOVE
    ex_heights = anchors[:, 3] - anchors[:, 1] + TO_REMOVE
    ex_ctr_x = anchors[:, 0] + 0.5 * ex_widths
    ex_ctr_y = anchors[:, 1] + 0.5 * ex_heights

    gt_widths = matched[:, 2] - matched[:, 0] + TO_REMOVE
    gt_heights = matched[:, 3] - matched[:, 1] + TO_REMOVE
    gt_ctr_x = matched[:, 0] + 0.5 * gt_widths
    gt_ctr_y = matched[:, 1] + 0.5 * gt_heights

    
    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths / variances[0]
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights / variances[0]
    targets_dw = torch.log(gt_widths / ex_widths) / variances[1]
    targets_dh = torch.log(gt_heights / ex_heights) / variances[1]

    targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)

    return targets

def decode(loc, anchors, variances=[0.1, 0.2], bbox_xform_clip=math.log(1000. / 16)):
#     """
#     Decode locations from predictions using anchors to undo
#     the encoding we did for offset regression at train time.
#     Args:
#         loc (tensor): location predictions for loc layers,
#             Shape: [num_anchors,4]
#         anchors (tensor): anchor boxes in center-offset form.
#             Shape: [num_anchors,4].
#         variances: (list[float]) Variances of anchorboxes
#     Return:
#         decoded bounding box predictions
#     """
#     #pdb.set_trace()
    
    TO_REMOVE = 1 if anchors[0,2]>1 else 0 # TODO remove
    widths = anchors[:, 2] - anchors[:, 0] + TO_REMOVE
    heights = anchors[:, 3] - anchors[:, 1] + TO_REMOVE
    ctr_x = anchors[:, 0] + 0.5 * widths
    ctr_y = anchors[:, 1] + 0.5 * heights

    dx = loc[:, 0::4] * variances[0]
    dy = loc[:, 1::4] * variances[0]
    dw = loc[:, 2::4] * variances[1]
    dh = loc[:, 3::4] * variances[1]

    # Prevent sending too large values into torch.exp()
    dw = torch.clamp(dw, max=bbox_xform_clip)
    dh = torch.clamp(dh, max=bbox_xform_clip)

    pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
    pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
    pred_w = torch.exp(dw) * widths[:, None]
    pred_h = torch.exp(dh) * heights[:, None]

    pred_boxes = torch.zeros_like(loc)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2 (note: "- 1" is correct; don't be fooled by the asymmetry)
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w - TO_REMOVE
    # y2 (note: "- 1" is correct; don't be fooled by the asymmetry)
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h - TO_REMOVE

    return pred_boxes


# Adapted from https://github.com/Hakuyume/chainer-ssd
def decode_01(loc, anchors, variances):
    """Decode locations from predictions using anchors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_anchors,4]
        anchors (tensor): anchor boxes in center-offset form.
            Shape: [num_anchors,4].
        variances: (list[float]) Variances of anchorboxes
    Return:
        decoded bounding box predictions
    """
    #pdb.set_trace()
    boxes = torch.cat((
        anchors[:, :2] + loc[:, :2] * variances[0] * anchors[:, 2:],
        anchors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes
    
def decode_seq(loc, anchors, variances, seq_len):
    boxes = []
    #print('variances', variances)
    for s in range(seq_len):
        if s == 0:
            boxes = decode(loc[:, :4], anchors[:, :4], variances)
        else:
            boxes = torch.cat((boxes,decode(loc[:,s*4:(s+1)*4], anchors[:,s*4:(s+1)*4], variances)),1)

    return boxes


def log_sum_exp(x):
    """Utility function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        x (Variable(tensor)): conf_preds from conf layers
    """
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x-x_max), 1, keepdim=True)) + x_max



# def nms_pt(boxes, scores, overlap=0.5):
#     keep = torchvision.ops.nms(boxes, scores, overlap)
#     return keep
    # gpu_keep = torchvision.ops.nms(boxes_for_nms.to('cuda'), scores.to('cuda'), iou_threshold)

# Original author: Francisco Massa:
# https://github.com/fmassa/object-detection.torch
# Ported to PyTorch by Max deGroot (02/01/2017)
def nms(boxes, scores, overlap=0.5, top_k=20, use_old_code=False):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_anchors,4].
        scores: (tensor) The class predscores for the img, Shape:[num_anchors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_anchors.
    """
    if use_old_code:
        keep = scores.new(scores.size(0)).zero_().long()
        if boxes.numel() == 0:
            return keep, 0
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        area = torch.mul(x2 - x1, y2 - y1)
        v, idx = scores.sort(0)  # sort in ascending order
        # I = I[v >= 0.01]
        idx = idx[-top_k:]  # indices of the top-k largest vals
        xx1 = boxes.new()
        yy1 = boxes.new()
        xx2 = boxes.new()
        yy2 = boxes.new()
        w = boxes.new()
        h = boxes.new()

        # keep = torch.Tensor()
        count = 0
        while idx.numel() > 0:
            i = idx[-1]  # index of current largest val
            # keep.append(i)
            keep[count] = i
            count += 1
            if idx.size(0) == 1:
                break
            idx = idx[:-1]  # remove kept element from view
            # load bboxes of next highest vals
            torch.index_select(x1, 0, idx, out=xx1)
            torch.index_select(y1, 0, idx, out=yy1)
            torch.index_select(x2, 0, idx, out=xx2)
            torch.index_select(y2, 0, idx, out=yy2)
            # store element-wise max with next highest score
            xx1 = torch.clamp(xx1, min=x1[i])
            yy1 = torch.clamp(yy1, min=y1[i])
            xx2 = torch.clamp(xx2, max=x2[i])
            yy2 = torch.clamp(yy2, max=y2[i])
            w.resize_as_(xx2)
            h.resize_as_(yy2)
            w = xx2 - xx1
            h = yy2 - yy1
            # check sizes of xx1 and xx2.. after each iteration
            w = torch.clamp(w, min=0.0)
            h = torch.clamp(h, min=0.0)
            inter = w*h
            # IoU = i / (area(a) + area(b) - i)
            rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
            union = (rem_areas - inter) + area[i]
            IoU = inter/union  # store result in iou
            # keep only elements with an IoU <= overlap
            idx = idx[IoU.le(overlap)]
    else:
        keep = torchvision.ops.nms(boxes, scores, overlap)
        count = keep.shape[0]
    
    return keep, count
