# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn.functional as F
from torch import nn

from models.transformer.util import box_ops
from utils.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, accuracy_sigmoid, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from models.backbone_builder import build_backbone
from models.detr.matcher_ucf import build_matcher
from models.detr.segmentation import (dice_loss, sigmoid_focal_loss)
# from models.transformer.transformer import build_transformer, TransformerEncoderLayer, TransformerEncoder
# from models.transformer.transformer_refined import build_transformer, TransformerEncoderLayer, TransformerEncoder
from models.transformer.transformer_refined import build_transformer
from models.transformer.transformer_layers import TransformerEncoderLayer, TransformerEncoder
import numpy as np


class DETR(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries,
                 hidden_dim, temporal_length, aux_loss=False, generate_lfb=False,
                 backbone_name='CSN-152', ds_rate=1, last_stride=True):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.temporal_length = temporal_length
        self.num_queries = num_queries
        self.transformer = transformer
        self.avg = nn.AvgPool3d(kernel_size=(temporal_length, 1, 1))
        self.avg_s = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.query_embed = nn.Embedding(num_queries * temporal_length, hidden_dim)

        if "SWIN" in backbone_name:
            print("using swin")
            self.input_proj = nn.Conv3d(1024, hidden_dim, kernel_size=1)
            self.class_proj = nn.Conv3d(1024, hidden_dim, kernel_size=1)
        elif "SlowFast" in backbone_name:
            self.input_proj = nn.Conv3d(backbone.num_channels, hidden_dim, kernel_size=1)
            self.class_proj = nn.Conv3d(2048 + 512, hidden_dim, kernel_size=1)
        else:
            self.input_proj = nn.Conv3d(backbone.num_channels, hidden_dim, kernel_size=1)
            self.class_proj = nn.Conv3d(backbone.num_channels, hidden_dim, kernel_size=1)

        encoder_layer = TransformerEncoderLayer(hidden_dim, 8, 2048, 0.1, "relu", normalize_before=False)
        self.encoder = TransformerEncoder(encoder_layer, num_layers=1, norm=None)
        self.cross_attn = nn.MultiheadAttention(256, num_heads=8, dropout=0.1)

        self.class_embed_b = nn.Linear(2048, 2)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        self.class_fc = nn.Linear(hidden_dim, num_classes + 1)
        self.dropout = nn.Dropout(0.5)

        self.backbone = backbone
        self.aux_loss = aux_loss
        self.hidden_dim = hidden_dim
        self.is_swin = "SWIN" in backbone_name

        self.generate_lfb = generate_lfb

        self.last_stride = last_stride


    def freeze_params(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.transformer.parameters():
            param.requires_grad = False
        for param in self.query_embed.parameters():
            param.requires_grad = False
        for param in self.bbox_embed.parameters():
            param.requires_grad = False
        for param in self.input_proj.parameters():
            param.requires_grad = False
        for param in self.class_embed_b.parameters():
            param.requires_grad = False

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)

        # if self.generate_lfb: return self.forward_lfb(samples)
        if self.generate_lfb: return self.forward_lfb_locations(samples)
        features, pos, xt = self.backbone(samples)
        src, mask = features[-1].decompose()
        assert mask is not None
        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]
        outputs_class_b = self.class_embed_b(self.avg_s(xt).squeeze(-1).squeeze(-1).squeeze(-1))
        outputs_class_b = outputs_class_b.unsqueeze(0).repeat(6, 1, 1)

        lay_n, bs, nb, dim = hs.shape

        src_c = self.class_proj(xt)

        hs_t_agg = hs.contiguous().view(lay_n, bs, 1, nb, dim)

        src_flatten = src_c.view(1, bs, self.hidden_dim, -1).repeat(lay_n, 1, 1, 1).view(lay_n * bs, self.hidden_dim, -1).permute(2, 0, 1).contiguous()
        if not self.is_swin:
            src_flatten, _ = self.encoder(src_flatten, orig_shape=src_c.shape)

        hs_query = hs_t_agg.view(lay_n * bs, nb, dim).permute(1, 0, 2).contiguous()
        q_class = self.cross_attn(hs_query, src_flatten, src_flatten)[0]
        q_class = q_class.permute(1, 0, 2).contiguous().view(lay_n, bs, nb, self.hidden_dim)

        outputs_class = self.class_fc(self.dropout(q_class))
        outputs_coord = self.bbox_embed(hs).sigmoid()

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1], 'pred_logits_b': outputs_class_b[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord, outputs_class_b)

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_class_b):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.

        return [{'pred_logits': a, 'pred_boxes': b, 'pred_logits_b': c}
                for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_class_b[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, weight, num_classes, num_queries, matcher, weight_dict, eos_coef, losses, data_file,
                 evaluation=False):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.weight = weight
        self.evaluation = evaluation
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.data_file = data_file

        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)

        src_logits_b = outputs['pred_logits_b']
        target_classes_b = torch.cat([t['vis'] for t in targets]).view(-1)
        loss_ce_b = F.cross_entropy(src_logits_b, target_classes_b)

        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        # target_classes[idx] = torch.clamp(target_classes_o,0,23)

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}
        losses['loss_ce_b'] = loss_ce_b

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]

        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """

        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes_list = []
        count = 0
        for t, (_, i) in zip(targets, indices):
            if int(t['vis']):
                target_boxes_list.append(t['boxes'][i])
                count += len(t['boxes'])
            else:
                # target_boxes_list.append(src_boxes[count])
                count += 1
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_boxes = target_boxes[:, 1:]

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        if num_boxes > 0:
            losses['loss_bbox'] = loss_bbox.sum() / num_boxes

            loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes),
                box_ops.box_cxcywh_to_xyxy(target_boxes)))
            losses['loss_giou'] = loss_giou.sum() / num_boxes
        else:
            loss_dumy = F.l1_loss(torch.as_tensor([1]).to(targets[0]["key_pos"].device), torch.as_tensor([1]).to(targets[0]["key_pos"].device), reduction='none')
            losses['loss_bbox'] = loss_dumy
            losses['loss_giou'] = loss_dumy
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"]

        # docs use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets]).decompose()
        target_masks = target_masks.to(src_masks)

        src_masks = src_masks[src_idx]
        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks[tgt_idx].flatten(1)

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        key_frames = torch.from_numpy(np.array([[self.num_queries * (targets[i]["key_pos"].cpu()) + j for j in range(self.num_queries)] for i in range(len(targets))]))
        key_frames = key_frames.view(len(key_frames), self.num_queries, 1).to(targets[0]["key_pos"].device)
        outputs_without_aux = {k: v.gather(1, key_frames.repeat(1, 1, v.shape[-1])) if k in ['pred_boxes', 'pred_logits'] else v for k, v in outputs.items() if k != 'aux_outputs'}
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)
        # _, sidx = self._get_src_permutation_idx(indices)
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs_without_aux, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs_ in enumerate(outputs['aux_outputs']):
                aux_outputs = {k: v.gather(1, key_frames.repeat(1, 1, v.shape[-1])) if k in ['pred_boxes', 'pred_logits'] else v for k, v in aux_outputs_.items() if k != 'aux_outputs'}
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """

        out_logits, out_bbox, out_logits_b = outputs['pred_logits'], outputs['pred_boxes'], outputs['pred_logits_b']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        scores = prob.detach().cpu().numpy()
        # labels = labels.detach().cpu().numpy()
        boxes = boxes.detach().cpu().numpy()

        output_b = out_logits_b.softmax(-1).detach().cpu().numpy()[..., 1:]

        return scores, boxes, output_b


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build_model(cfg):
    num_classes = cfg.CONFIG.DATA.NUM_CLASSES
    print('num_classes', num_classes)

    backbone = build_backbone(cfg)
    transformer = build_transformer(cfg)

    model = DETR(backbone,
                 transformer,
                 num_classes=cfg.CONFIG.DATA.NUM_CLASSES,
                 num_queries=cfg.CONFIG.MODEL.QUERY_NUM * cfg.CONFIG.MODEL.TEMP_LEN // cfg.CONFIG.MODEL.DS_RATE if not cfg.CONFIG.MODEL.SINGLE_FRAME else cfg.CONFIG.MODEL.QUERY_NUM,
                 aux_loss=cfg.CONFIG.TRAIN.AUX_LOSS,
                 hidden_dim=cfg.CONFIG.MODEL.D_MODEL,
                 temporal_length=cfg.CONFIG.MODEL.TEMP_LEN,
                 generate_lfb=cfg.CONFIG.MODEL.GENERATE_LFB,
                 backbone_name=cfg.CONFIG.MODEL.BACKBONE_NAME,
                 ds_rate=cfg.CONFIG.MODEL.DS_RATE,
                 last_stride=cfg.CONFIG.MODEL.LAST_STRIDE)

    matcher = build_matcher(cfg)
    weight_dict = {'loss_ce': cfg.CONFIG.LOSS_COFS.DICE_COF, 'loss_bbox': cfg.CONFIG.LOSS_COFS.BBOX_COF}
    weight_dict['loss_giou'] = cfg.CONFIG.LOSS_COFS.GIOU_COF
    weight_dict['loss_ce_b'] = 1

    if cfg.CONFIG.TRAIN.AUX_LOSS:
        aux_weight_dict = {}
        for i in range(cfg.CONFIG.MODEL.DEC_LAYERS - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes'] #, 'cardinality'
    criterion = SetCriterion(cfg.CONFIG.LOSS_COFS.WEIGHT,
                             num_classes,
                             num_queries=cfg.CONFIG.MODEL.QUERY_NUM,
                             matcher=matcher, weight_dict=weight_dict,
                             eos_coef=cfg.CONFIG.LOSS_COFS.EOS_COF,
                             losses=losses,
                             data_file=cfg.CONFIG.DATA.DATASET_NAME,
                             evaluation=cfg.CONFIG.EVAL_ONLY)
    postprocessors = {'bbox': PostProcess()}

    return model, criterion, postprocessors

