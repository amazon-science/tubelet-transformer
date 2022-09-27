import numpy as np
import pdb
from modules import utils
import scipy.signal as signal
logger = utils.get_logger(__name__)
from scipy.signal import savgol_filter
# from gen_dets import make_joint_probs_from_marginals
from modules.utils import make_joint_probs_from_marginals

over_s = 0.0
under_s = 0.0
over_e = 0.0
under_e = 0.0
oa_s = 0.0
ua_s = 0.0
oa_e = 0.0
ua_e = 0.0

def make_det_tube(scores, boxes, frames, label_id):
    tube = {}
    tube['label_id'] =label_id
    tube['scores'] = np.asarray(scores)
    tube['boxes'] = np.asarray(boxes)
    tube['score'] = np.mean(scores)
    tube['frames'] = np.asarray(frames)
    # assert tube['frames'].shape[0] == tube['boxes'].shape[0], 'must be equal'
    return tube

def get_nonnp_det_tube(scores, boxes, start, end, label_id, score=None):
    tube = {}
    tube['label_id'] =label_id
    tube['scores'] = scores
    tube['boxes'] = boxes
    
    if score is not None:
        tube['score'] = score
    else:
        tube['score'] = float(np.mean(scores))

    tube['frames'] = np.asarray([i for i in range(start, end)])
    assert len(tube['frames']) == len(tube['boxes']), 'must be equal'

    return tube

def make_gt_tube(frames, boxes, label_id):
    frames = np.asarray(frames)
    indexs = np.argsort(frames)
    frames = frames[indexs]
    boxes = np.asarray(boxes)
    if boxes.shape[0]>0:
        boxes = boxes[indexs,:]
    tube = {}
    tube['frames'] = frames
    tube['boxes'] = boxes
    tube['label_id'] = label_id
    return tube

def trim_tubes(start_id, numc, paths, childs, num_classes_list, topk=5, alpha=3, min_len=3, trim_method='None'):
    """ Trim the paths into tubes using DP"""
    tubes = []
    for path in paths:
        if len(childs)>0:
            allScores = make_joint_probs_from_marginals(path['allScores'], childs, num_classes_list, start_id=0)
        else:
            allScores = path['allScores']
        allScores = allScores[:,start_id:start_id+numc]
        path_start_frame = path['foundAt'][0]
        if allScores.shape[0]<=min_len:
            continue
        
        # print(allScores.shape)
        if trim_method == 'none': # 
            # print('no trimming')
            topk_classes, topk_scores = get_topk_classes(allScores, topk)
            for i in range(topk):
                label, start, end = topk_classes[i], path_start_frame, allScores.shape[0] + path_start_frame 
                if end-start+1 > min_len:
                    # tube = get_nonnp_det_tube(allScores[:,label], path['boxes'], int(start), int(end), int(label))
                    tube = get_nonnp_det_tube(allScores[:,label], path['boxes'], int(start), int(end), int(label), score=topk_scores[i])
                    tubes.append(tube)
        elif trim_method == 'dpscores': ## standarded method Multi class-DP
            allScores = path['allScores'][:,start_id:start_id+numc]
            score_mat = np.transpose(allScores.copy())
            for _ in range(topk):
                (segments, _) = dpEMmax(score_mat, alpha)
                # print(segments)
                labels, starts, ends = getLabels(segments)
                # print(labels, starts, ends)
                for i in range(len(labels)):
                    if ends[i] - starts[i] >= min_len:
                        scores = score_mat[labels[i], starts[i]:ends[i]+1]
                        boxes = path['boxes'][starts[i]:ends[i]+1, :]
                        start = starts[i] + path_start_frame
                        end = ends[i] + path_start_frame + 1
                        tube = get_nonnp_det_tube(scores, boxes, int(start), int(end), int(labels[i]))
                        tubes.append(tube)
                        score_mat[labels[i], starts[i]:ends[i]+1] = 0.0

        elif trim_method == 'dpscorestopn': ## bit fancy only select top segments
            score_mat = np.transpose(allScores.copy())
            for _ in range(topk):
                (segments, _) = dpEMmax(score_mat, alpha)
                # print(segments)
                labels, starts, ends = getLabels(segments)
                # print(labels, starts, ends)
                num_seg = labels.shape[0]
                seg_scores = np.zeros(num_seg)
                for i in range(min(2,len(labels))):
                    if ends[i] - starts[i] >= min_len:
                        scores = score_mat[labels[i], starts[i]:ends[i]+1]
                        seg_scores[i] = np.mean(scores)
                    else:
                        score_mat[labels[i], starts[i]:ends[i]+1] = 0.0
                        seg_scores[i] = 0.0

                inds = np.argsort(-seg_scores)
                for ii in range(min(2, num_seg)):
                    i = inds[ii]
                    # if ends[i] - starts[i] >= min_len:
                    scores = score_mat[labels[i], starts[i]:ends[i]+1]
                    boxes = path['boxes'][starts[i]:ends[i]+1, :]
                    start = starts[i] + path_start_frame
                    if boxes.shape[0] != -starts[i] + ends[i] + 1:
                        print('We have exceptions', boxes.shape[0], -starts[i] + ends[i]+1)
                    end = ends[i] + path_start_frame + 1
                    tube = get_nonnp_det_tube(scores, boxes, int(start), int(end), int(labels[i]))
                    tubes.append(tube)
                    score_mat[labels[i], starts[i]:ends[i]+1] = 0.0
        else: #indvidual class-wise dp
            aa = 0
            if alpha == 0 and numc == 24:
                # alphas = [1, 1, 16, 1, 1, 2, 16, 8,  4, 16, 6, 16, 20, 16, 1, 16, 16, 20, 16, 2, 4, 8, 1, 20]
                # alphas = [1, 1,  8, 1, 1, 3, 16, 16, 2, 16, 3, 16, 20, 16, 1,  8,  8,  8, 16, 2, 2, 8, 1, 20]
                # alphas = [1, 5, 16, 8, 1, 3, 16, 16, 16, 3, 8, 16, 16, 16, 1, 5, 16, 16, 5, 2, 1, 8, 3, 16]
                # alphas = [1, 3, 16, 2, 1, 3, 8, 16, 16, 3, 3, 16, 16, 16, 1, 5, 16, 8, 5, 2, 1, 16, 2, 16]
                alphas = [1, 1, 16, 3, 1, 8, 16, 16, 10, 10, 3, 16, 16, 10, 1, 8, 16, 16, 16, 2, 1, 8, 2, 16]
            else:
                alphas = np.zeros(numc)+alpha
                
            topk_classes, topk_scores = get_topk_classes(allScores, topk)
            for idx in range(topk_classes.shape[0]):
                current_label = int(topk_classes[idx])
                if numc == 24:
                    in_scores = path['allScores'][:,start_id-1]
                else:
                    in_scores = allScores[:,current_label]

                smooth_scores = signal.medfilt(in_scores, 5)
                smooth_scores = in_scores/np.max(smooth_scores)
                score_mat =  np.hstack((smooth_scores[:, np.newaxis], 1 - smooth_scores[:, np.newaxis])) 
                score_mat = np.transpose(score_mat.copy())
                (segments, _) = dpEMmax(score_mat, alphas[current_label])
                labels, starts, ends = getLabels(segments)
                for i in range(len(labels)):
                    if ends[i] - starts[i] >= min_len and labels[i]==0:
                        scores = allScores[starts[i]:ends[i]+1, current_label]
                        sorted_classes = np.argsort(-scores)
                        sorted_scores = scores[sorted_classes]
                        topn = max(1,int(sorted_scores.shape[0]/2))
                        mscore = np.mean(sorted_scores[:topn])
                        boxes = path['boxes'][starts[i]:ends[i]+1, :]
                        start = starts[i] + path_start_frame
                        end = ends[i] + path_start_frame + 1
                        sf = max(1,int(start)-aa)
                        ef = int(end)-(start-sf)
                        tube = get_nonnp_det_tube(scores, boxes, sf, ef, int(current_label), score=mscore) #topk_scores[idx])
                        tubes.append(tube)
                        # score_mat[labels[i], starts[i]:ends[i]+1] = 0.0
    return tubes

def getLabels(segments, cls=1):
    starts = np.zeros(len(segments), dtype='int32')
    ends = np.zeros(len(segments), dtype='int32')
    labels = np.zeros(len(segments), dtype='int32')
    fl = 0
    i=0
    starts[i]=0
    fl = segments[0]
    labels[i] =  segments[0]
#    print segments[0]
#    pdb.set_trace()
    for ii in range(len(segments)):
        if abs(segments[ii] -fl)>0:
            ends[i]=ii-1
            fl = segments[ii]
            i+=1
            starts[i]=ii
            labels[i] = fl
    ends[i] = len(segments)-1
    return labels[:i+1],starts[:i+1],ends[:i+1]

def get_topk_classes(allScores, topk):
    scores = np.zeros(allScores.shape[1])
    # print(scores.shape)
    topn = max(1, allScores.shape[1]//4)
    for k in range(scores.shape[0]):
        temp_scores = allScores[:,k]
        sorted_score = np.sort(-temp_scores)
        # print(sorted_score[:topn])
        scores[k] = np.mean(-sorted_score[:topn])
    sorted_classes = np.argsort(-scores)
    sorted_scores = scores[sorted_classes]
    # sorted_scores = sorted_scores/np.sum(sorted_scores)
    # print(sorted_scores)
    return sorted_classes[:topk], sorted_scores[:topk]


def dpEMmax(M, alpha=3):
    (r,c) = np.shape(M)
    D = np.zeros((r, c+1)) # add an extra column
    D[:,0] = 1 # % put the maximum cost
    D[:, 1:(c+1)] = M
    phi = np.zeros((r,c))
    for j in range(1,c):
        for i in range(r):
            v1 = np.ones(r)*alpha
            v1[i] = 0
            values= D[:, j-1] - v1
            tb = np.argmax(values)
            dmax = max(values)
            D[i,j] = D[i,j]+dmax
            phi[i,j] = tb

    q = c-1
    values= D[:, c-1]
    p = np.argmax(values)
    i = p
    j = q 
    ps = np.zeros(c)
    ps[q] = p
    while j>0:
        tb = phi[i,j]
        j = int(j-1)
        q = j
        ps[q] = tb
        i = int(tb)
    
    D = D[:,1:]
    return (ps,D)


def intersect(box_a, box_b):
        # A = box_a.size(0)
        B = box_b.shape[0]
        inters = np.zeros(B)
        for b in range(B):
            max_x = min(box_a[2], box_b[b, 2])
            max_y = min(box_a[3], box_b[b, 3])
            min_x = max(box_a[0], box_b[b, 0])
            min_y = max(box_a[1], box_b[b, 1])
            inters[b] = (max_x-min_x)*(max_y-min_y)
        return inters


def bbox_overlaps(box_a, box_b):

    inter = intersect(box_a, box_b)
    area_a = (box_a[2]-box_a[0])*(box_a[3]-box_a[1])
    B = box_b.shape[0]
    ious = np.zeros(B)
    for b in range(B):
        if inter[b]>0:
            area_b = (box_b[b,2] - box_b[b,0]) * (box_b[b,3] - box_b[b,1])
            union = area_a + area_b - inter[b]
            ious[b] = inter[b]/union
    return ious


def get_tube_3Diou(tube_a, tube_b , metric_type='stiou'):
    """Compute the spatio-temporal IoU between two tubes"""

    

    tmin = max(tube_a['frames'][0], tube_b['frames'][0])
    tmax = min(tube_a['frames'][-1], tube_b['frames'][-1])
    
    if tmax < tmin: return 0.0

    temporal_inter = tmax - tmin + 1
    temporal_union = max(tube_a['frames'][-1], tube_b['frames'][-1]) - min(tube_a['frames'][0], tube_b['frames'][0]) + 1
    tiou = temporal_inter / temporal_union
    if metric_type == 'tiou':
        return tiou
    # try:

    tube_a_boxes = tube_a['boxes'][int(np.where(tube_a['frames'] == tmin)[0][0]): int(
        np.where(tube_a['frames'] == tmax)[0][0]) + 1, :]
    tube_b_boxes = tube_b['boxes'][int(np.where(tube_b['frames'] == tmin)[0][0]): int(
        np.where(tube_b['frames'] == tmax)[0][0]) + 1, :]
    # except:
    #     pdb.set_trace()     print('something', tube_a_boxes, tube_b_boxes, iou)

    siou = iou3d(tube_a_boxes, tube_b_boxes)

    global over_s, over_e, under_s, under_e, oa_s, oa_e, ua_s, ua_e
    
    if tube_a['frames'][-1]>= tube_b['frames'][-1]:
        over_e += 1
        oa_e += tube_a['frames'][-1] - tube_b['frames'][-1] 
    else:
        under_e += 1
        ua_e += tube_a['frames'][-1] - tube_b['frames'][-1]
    
    if tube_a['frames'][0]<= tube_b['frames'][0]:
        over_s += 1
        oa_s += tube_a['frames'][0] - tube_b['frames'][0] 
    else:
        under_s += 1
        ua_s += tube_a['frames'][0] - tube_b['frames'][0]
    
    # if not (tube_a['frames'][-1]>= tube_b['frames'][-1] and tube_a['frames'][0]<= tube_b['frames'][0]):
    #     tiou = 1.0
    # logger.info('over_s {} over_e {} under_s {} under_e {} oa_s {} oa_e {} ua_s {} ua_e {}'.format(over_s, over_e, under_s, under_e, oa_s, oa_e, ua_s, ua_e))
    # if siou>0.5 and temporal_inter>= tube_b['frames'][-1]-tube_b['frames'][0]:
    #     print(tube_b['frames'][0],tube_b['frames'][-1], tube_a['frames'][0],tube_a['frames'][-1], tube_a['scores'])
    if metric_type == 'siou':
        return siou
    else:
        return  siou * tiou


def iou3d(tube_a, tube_b):
    """Compute the IoU between two tubes with same temporal extent"""

    assert tube_a.shape[0] == tube_b.shape[0]
    # assert np.all(b1[:, 0] == b2[:, 0])

    ov = overlap2d(tube_a,tube_b)

    return np.mean(ov / (area2d(tube_a) + area2d(tube_b) - ov) )


def area2d(b):
    """Compute the areas for a set of 2D boxes"""

    return (b[:,2]-b[:,0]+1) * (b[:,3]-b[:,1]+1)


def overlap2d(b1, b2):
    """Compute the overlaps between a set of boxes b1 and one box b2"""

    xmin = np.maximum(b1[:,0], b2[:,0])
    ymin = np.maximum(b1[:,1], b2[:,1])
    xmax = np.minimum(b1[:,2] + 1, b2[:,2] + 1)
    ymax = np.minimum(b1[:,3] + 1, b2[:,3] + 1)

    width = np.maximum(0, xmax - xmin)
    height = np.maximum(0, ymax - ymin)

    return width * height

def nms3dt(tubes, overlap=0.6):
    """Compute NMS of scored tubes. Tubes are given as list of (tube, score)
    return the list of indices to keep
    """

    if not tubes:
        return np.array([], dtype=np.int32)

    I = np.argsort([t['score'] for t in tubes])
    indices = np.zeros(I.size, dtype=np.int32)
    counter = 0

    while I.size > 0:
        i = I[-1]
        indices[counter] = i
        counter += 1
        ious = np.array([get_tube_3Diou(tubes[ii], tubes[i]) for ii in I[:-1]])
        I = I[np.where(ious <= overlap)[0]]
    indices = indices[:counter]
    final_tubes = []
    for ind in indices:
        final_tubes.append(tubes[ind])
    
    return final_tubes
