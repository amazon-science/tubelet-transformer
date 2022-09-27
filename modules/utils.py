import os, sys
import shutil
import socket
import getpass
import copy
import numpy as np
from modules.box_utils import nms
import datetime
import logging 
import torch
import pdb
import torchvision

# from https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
class BufferList(torch.nn.Module):
    """    
    Similar to nn.ParameterList, but for buffers
    """

    def __init__(self, buffers=None):
        super(BufferList, self).__init__()
        if buffers is not None:
            self.extend(buffers)

    def extend(self, buffers):
        offset = len(self)
        for i, buffer in enumerate(buffers):
            self.register_buffer(str(offset + i), buffer)
        return self

    def __len__(self):
        return len(self._buffers)

    def __iter__(self):
        return iter(self._buffers.values())
        
def setup_logger(args):
    """
    Sets up the logging.
    """
    log_file_name = '{:s}/{:s}-{date:%m-%d-%Hx}.log'.format(args.SAVE_ROOT, args.MODE, date=datetime.datetime.now())
    args.log_dir = 'logs/'+args.exp_name+'/'
    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)
        
    added_log_file = '{}{}-{date:%m-%d-%Hx}.log'.format(args.log_dir, args.MODE, date=datetime.datetime.now())

   
    # Set up logging format.
    _FORMAT = "[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s"

    logging.root.handlers = []
    logging.basicConfig(
        level=logging.INFO, format=_FORMAT, stream=sys.stdout
    )
    logging.getLogger().addHandler(logging.FileHandler(log_file_name, mode='a'))
    # logging.getLogger().addHandler(logging.FileHandler(added_log_file, mode='a'))


def get_logger(name):
    """
    Retrieve the logger with the specified name or, if name is None, return a
    logger which is the root logger of the hierarchy.
    Args:
        name (string): name of the logger.
    """
    return logging.getLogger(name)

def copy_source(source_dir):
    if not os.path.isdir(source_dir):
        os.system('mkdir -p ' + source_dir)
    
    for dirpath, dirs, files in os.walk('./', topdown=True):
        for file in files:
            if file.endswith('.py'): #fnmatch.filter(files, filepattern):
                shutil.copy2(os.path.join(dirpath, file), source_dir)


def set_args(args):
    args.MAX_SIZE = int(args.MIN_SIZE*1.35)
    args.MILESTONES = [int(val) for val in args.MILESTONES.split(',')]
    #args.GAMMAS = [float(val) for val in args.GAMMAS.split(',')]
    args.EVAL_EPOCHS = [int(val) for val in args.EVAL_EPOCHS.split(',')]

    args.TRAIN_SUBSETS = [val for val in args.TRAIN_SUBSETS.split(',') if len(val)>1]
    args.VAL_SUBSETS = [val for val in args.VAL_SUBSETS.split(',') if len(val)>1]
    args.TEST_SUBSETS = [val for val in args.TEST_SUBSETS.split(',') if len(val)>1]
    args.TUBES_EVAL_THRESHS = [ float(val) for val in args.TUBES_EVAL_THRESHS.split(',') if len(val)>0.0001]
    args.model_subtype = args.MODEL_TYPE.split('-')[0]
    ## check if subsets are okay
    possible_subets = ['test', 'train','val']
    for idx in range(1,4):
        possible_subets.append('train_'+str(idx))        
        possible_subets.append('val_'+str(idx))        

    if len(args.VAL_SUBSETS) < 1 and args.DATASET == 'road':
        args.VAL_SUBSETS = [ss.replace('train', 'val') for ss in args.TRAIN_SUBSETS]
    if len(args.TEST_SUBSETS) < 1:
        # args.TEST_SUBSETS = [ss.replace('train', 'val') for ss in args.TRAIN_SUBSETS]
        args.TEST_SUBSETS = args.VAL_SUBSETS
    
    for subsets in [args.TRAIN_SUBSETS, args.VAL_SUBSETS, args.TEST_SUBSETS]:
        for subset in subsets:
            assert subset in possible_subets, 'subest should from one of these '+''.join(possible_subets)

    args.DATASET = args.DATASET.lower()
    args.ARCH = args.ARCH.lower()

    args.MEANS =[0.485, 0.456, 0.406]
    args.STDS = [0.229, 0.224, 0.225]

    username = getpass.getuser()
    hostname = socket.gethostname()
    args.hostname = hostname
    args.user = username
    
    args.model_init = 'kinetics'

    args.MODEL_PATH = args.MODEL_PATH[:-1] if args.MODEL_PATH.endswith('/') else args.MODEL_PATH 

    assert args.MODEL_PATH.endswith('kinetics-pt') or args.MODEL_PATH.endswith('imagenet-pt') 
    args.model_init = 'imagenet' if args.MODEL_PATH.endswith('imagenet-pt') else 'kinetics'
    
    if args.MODEL_PATH == 'imagenet':
        args.MODEL_PATH = os.path.join(args.MODEL_PATH, args.ARCH+'.pth')
    else:
        args.MODEL_PATH = os.path.join(args.MODEL_PATH, args.ARCH+args.MODEL_TYPE+'.pth')
            
    
    print('Your working directories are::\nLOAD::> ', args.DATA_ROOT, '\nSAVE::> ', args.SAVE_ROOT)
    print('Your model will be initialized using', args.MODEL_PATH)
    
    return args


def create_exp_name(args):
    """Create name of experiment using training parameters """
    splits = ''.join([split[0]+split[-1] for split in args.TRAIN_SUBSETS])
    args.exp_name = '{:s}{:s}{:d}-P{:s}-b{:0d}s{:d}x{:d}x{:d}-{:s}{:s}-h{:d}x{:d}x{:d}'.format(
        args.ARCH, args.MODEL_TYPE,
        args.MIN_SIZE, args.model_init, args.BATCH_SIZE,
        args.SEQ_LEN, args.MIN_SEQ_STEP, args.MAX_SEQ_STEP,
        args.DATASET, splits, 
        args.HEAD_LAYERS, args.CLS_HEAD_TIME_SIZE,
        args.REG_HEAD_TIME_SIZE,
        )

    args.SAVE_ROOT += args.DATASET+'/'
    args.SAVE_ROOT = args.SAVE_ROOT+'cache/'+args.exp_name+'/'
    if not os.path.isdir(args.SAVE_ROOT):
        print('Create: ', args.SAVE_ROOT)
        os.makedirs(args.SAVE_ROOT)

    return args
    
# Freeze batch normlisation layers
def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') > -1:
        m.eval()
        if m.affine:
            m.weight.requires_grad = False
            m.bias.requires_grad = False
        

def get_individual_labels(gt_boxes, tgt_labels):
    # print(gt_boxes.shape, tgt_labels.shape)
    new_gts = np.zeros((gt_boxes.shape[0]*20, 5))
    ccc = 0
    for n in range(tgt_labels.shape[0]):
        for t in range(tgt_labels.shape[1]):
            if tgt_labels[n,t]>0:
                new_gts[ccc, :4] = gt_boxes[n,:]
                new_gts[ccc, 4] = t
                ccc += 1
    return new_gts[:ccc,:]


def get_individual_location_labels(gt_boxes, tgt_labels):
    return [gt_boxes, tgt_labels]


def filter_detections(args, scores, decoded_boxes_batch):
    c_mask = scores.gt(args.CONF_THRESH)  # greater than minmum threshold
    scores = scores[c_mask].squeeze()
    if scores.dim() == 0 or scores.shape[0] == 0:
        return np.asarray([])
    
    boxes = decoded_boxes_batch[c_mask, :].view(-1, 4)
    ids, counts = nms(boxes, scores, args.NMS_THRESH, args.TOPK*5)  # idsn - ids after nms
    scores = scores[ids[:min(args.TOPK,counts)]].cpu().numpy()
    boxes = boxes[ids[:min(args.TOPK,counts)]].cpu().numpy()
    cls_dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=True)

    return cls_dets


def filter_detections_for_tubing(args, scores, decoded_boxes_batch, confidences):
    c_mask = scores.gt(args.CONF_THRESH)  # greater than minmum threshold
    scores = scores[c_mask].squeeze()
    if scores.dim() == 0 or scores.shape[0] == 0:
        return  np.zeros((0,200))
    
    boxes = decoded_boxes_batch[c_mask, :].clone().view(-1, 4)
    numc = confidences.shape[-1]
    confidences = confidences[c_mask,:].clone().view(-1, numc)

    max_k = min(args.TOPK*60, scores.shape[0])
    ids, counts = nms(boxes, scores, args.NMS_THRESH, max_k)  # idsn - ids after nms
    scores = scores[ids[:min(args.TOPK,counts)]].cpu().numpy()
    boxes = boxes[ids[:min(args.TOPK,counts)],:].cpu().numpy()
    confidences = confidences[ids[:min(args.TOPK, counts)],:].cpu().numpy()
    cls_dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=True)
    save_data = np.hstack((cls_dets, confidences[:,1:])).astype(np.float32)
    #print(save_data.shape)
    return save_data


def filter_detections_for_dumping(args, scores, decoded_boxes_batch, confidences):
    c_mask = scores.gt(args.GEN_CONF_THRESH)  # greater than minmum threshold
    scores = scores[c_mask].squeeze()
    if scores.dim() == 0 or scores.shape[0] == 0:
        return np.zeros((0,5)), np.zeros((0,200))
    
    boxes = decoded_boxes_batch[c_mask, :].clone().view(-1, 4)
    numc = confidences.shape[-1]
    confidences = confidences[c_mask,:].clone().view(-1, numc)

    # sorted_ind = np.argsort(-scores.cpu().numpy())
    # sorted_ind = sorted_ind[:topk*10]
    # boxes_np = boxes.cpu().numpy()
    # confidences_np = confidences.cpu().numpy()
    # save_data = np.hstack((boxes_np[sorted_ind,:], confidences_np[sorted_ind, :]))
    # args.GEN_TOPK, args.GEN_NMS
     
    max_k = min(args.GEN_TOPK*500, scores.shape[0])
    ids, counts = nms(boxes, scores, args.GEN_NMS, max_k)  # idsn - ids after nms
    # keepids = torchvision.ops.nms(boxes, scores, args.GEN_NMS)
    # pdb.set_trace()
    scores = scores[ids[:min(args.GEN_TOPK,counts)]].cpu().numpy()
    boxes = boxes[ids[:min(args.GEN_TOPK,counts)],:].cpu().numpy()
    confidences = confidences[ids[:min(args.GEN_TOPK, counts)],:].cpu().numpy()
    cls_dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=True)
    save_data = np.hstack((cls_dets, confidences[:,1:])).astype(np.float32)
    #print(save_data.shape)
    return cls_dets, save_data

def make_joint_probs_from_marginals(frame_dets, childs, num_classes_list, start_id=4):
    
    # pdb.set_trace()

    add_list = copy.deepcopy(num_classes_list[:3])
    add_list[0] = start_id+1
    add_list[1] = add_list[0]+add_list[1]
    add_list[2] = add_list[1]+add_list[2]
    # for ind in range(frame_dets.shape[0]):
    for nlt, ltype in enumerate(['duplex','triplet']):
        lchilds = childs[ltype+'_childs']
        lstart = start_id
        for num in num_classes_list[:4+nlt]:
            lstart += num
        
        for c in range(num_classes_list[4+nlt]):
            tmp_scores = []
            for chid, ch in enumerate(lchilds[c]):
                if len(tmp_scores)<1:
                    tmp_scores = copy.deepcopy(frame_dets[:,add_list[chid]+ch])
                else:
                    tmp_scores *= frame_dets[:,add_list[chid]+ch]
            frame_dets[:,lstart+c] = tmp_scores

    return frame_dets



def eval_strings():
    return ["Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = ",
            "Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = ",
            "Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = ",    
            "Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = ",
            "Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = ",
            "Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = ",
            "Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = ",
            "Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = ",
            "Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = ",
            "Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = ",
            "Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = ",
            "Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = "]
