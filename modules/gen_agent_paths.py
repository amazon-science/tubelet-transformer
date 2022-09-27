import numpy as np
import pdb

def update_agent_paths(live_paths, dead_paths, dets, num_classes_to_use, time_stamp, iouth=0.1, costtype='scoreiou', jumpgap=5, min_len=5): ## trim_threshold=100, keep_num=60,
    num_box = dets['boxes'].shape[0]
    if len(live_paths) == 0:
        # Start a path for each box in first frame
        for b in range(num_box):
            live_paths.append({'boxes': None, 'scores': [], 'allScores': None, 'foundAt': [], 'count': 1})
            live_paths[b]['boxes'] = dets['boxes'][b, :].reshape(1,-1)  # bth box x0,y0,x1,y1 at frame t
            live_paths[b]['scores'].append(dets['scores'][b])  # action score of bth box at frame t
            live_paths[b]['allScores'] = dets['allScores'][b, :].reshape(1,-1)  # scores for all action for bth box at frame t
            live_paths[b]['foundAt'].append(time_stamp)  # frame box was found in
            live_paths[b]['count'] = 1  # current box count for bth box tube
    else:
        # Link each path to detections at frame t
        lp_count = len(live_paths)  # total paths at time t
        dead_count = 0
        covered_boxes = np.zeros(num_box)
        path_order_score = np.zeros(lp_count)
        avoid_dets = []
        for lp in range(lp_count):
            # Check whether path has gone stale
            if time_stamp - live_paths[lp]['foundAt'][-1] <= jumpgap:
                # IoU scores for path lp
                as1 = live_paths[lp]['allScores'][-1,:num_classes_to_use]
                as2 = dets['allScores'][:,:num_classes_to_use]
                box_to_lp_score = score_of_edge(live_paths[lp], dets, iouth, costtype, avoid_dets, as1, as2, jumpgap)
                
                if np.sum(box_to_lp_score) > 0.1: 
                    # print('We are here', np.sum(box_to_lp_score)) 
                    # check if there's at least one match to detection in this frame
                    maxInd = np.argmax(box_to_lp_score)
                    # m_score = np.max(box_to_lp_score)
                    live_paths[lp]['count'] = live_paths[lp]['count'] + 1
                    live_paths[lp]['boxes'] = np.vstack((live_paths[lp]['boxes'], dets['boxes'][maxInd, :]))
                    live_paths[lp]['scores'].append(dets['scores'][maxInd])
                    live_paths[lp]['allScores'] = np.vstack((live_paths[lp]['allScores'], dets['allScores'][maxInd, :]))
                    live_paths[lp]['foundAt'].append(time_stamp)
                    avoid_dets.append(maxInd)
                    covered_boxes[maxInd] = 1
                
                # else:
                # live_paths[lp]['lastfound'] += 1
                scores = sorted(np.asarray(live_paths[lp]['scores']))
                num_sc = len(scores)
                path_order_score[lp] = np.mean(np.asarray(scores[int(max(0, num_sc - jumpgap-1)):num_sc]))
            else:
                # If the path is stale, increment the dead_count
                dead_count += 1
        
        # Sort the path based on score of the boxes and terminate dead path
        if len(path_order_score)>1 or dead_count>0:
            # print('sorting path')
            live_paths, dead_paths = sort_live_paths(live_paths, path_order_score, dead_paths, jumpgap, time_stamp)


        # start new paths using boxes that are not assigned
        lp_count = len(live_paths)
        if np.sum(covered_boxes) < num_box:
            for b in range(num_box):
                if covered_boxes[b] < 0.99:
                    # print('numb and covered ', num_box, covered_boxes)
                    live_paths.append({'boxes': [], 'scores': [], 'allScores': None, 'foundAt': [], 'count': 1})
                    live_paths[lp_count]['boxes'] = dets['boxes'][b, :].reshape(1,-1)  # bth box x0,y0,x1,y1 at frame t
                    live_paths[lp_count]['scores'].append(dets['scores'][b])  # action score of bth box at frame t
                    live_paths[lp_count]['allScores'] = dets['allScores'][b, :].reshape(1,-1)  # scores for all action for bth box at frame t
                    live_paths[lp_count]['count'] = 1  # current box count for bth box tube
                    live_paths[lp_count]['foundAt'].append(time_stamp)  # frame box was found in
                    lp_count += 1

    # live_paths = trim_paths(live_paths, trim_threshold, keep_num)
    # dead_paths = remove_dead_paths(dead_paths, min_len, time_stamp)

    return live_paths, dead_paths

def trim_paths(live_paths, trim_threshold, keep_num):
    lp_count = len(live_paths)
    for lp in range(lp_count):
        # print(live_paths[lp]['boxes'].shape, live_paths[lp]['allScores'].shape)
        if len(live_paths[lp]['boxes']) > trim_threshold:
            live_paths[lp]['boxes'] = live_paths[lp]['boxes'][-keep_num:, :]
            live_paths[lp]['scores'] = live_paths[lp]['scores'][-keep_num:]
            live_paths[lp]['allScores'] = live_paths[lp]['allScores'][-keep_num:, :]
            live_paths[lp]['foundAt'] = live_paths[lp]['foundAt'][-keep_num:]
    return live_paths


def remove_dead_paths(live_paths, min_len, time_stamp):
    dead_paths = []
    dp_count = 0
    for olp in range(len(dead_paths)):
        if len(dead_paths[olp]['boxes']) >= min_len:
            dead_paths.append({'boxes': None, 'scores': None, 'allScores': None,
                               'foundAt': None, 'count': None})
            dead_paths[dp_count]['boxes'] = live_paths[olp]['boxes']
            dead_paths[dp_count]['scores'] = live_paths[olp]['scores']
            dead_paths[dp_count]['allScores'] = live_paths[olp]['allScores']
            dead_paths[dp_count]['foundAt'] = live_paths[olp]['foundAt']
            dead_paths[dp_count]['count'] = live_paths[olp]['count']
            dp_count += 1

    return dead_paths

def sort_live_paths(live_paths, path_order_score, dead_paths, jumpgap, time_stamp):
    inds = path_order_score.flatten().argsort()[::-1]
    sorted_live_paths = []
    lpc = 0
    dp_count = len(dead_paths)
    for lp in range(len(live_paths)):
        olp = inds[lp]
        if time_stamp-live_paths[olp]['foundAt'][-1] <= jumpgap:
            sorted_live_paths.append({'boxes': None, 'scores': None, 'allScores': None,
                                      'foundAt': None, 'count': None})
            sorted_live_paths[lpc]['boxes'] = live_paths[olp]['boxes']
            sorted_live_paths[lpc]['scores'] = live_paths[olp]['scores']
            sorted_live_paths[lpc]['allScores'] = live_paths[olp]['allScores']
            sorted_live_paths[lpc]['foundAt'] = live_paths[olp]['foundAt']
            sorted_live_paths[lpc]['count'] = live_paths[olp]['count']
            lpc += 1
        else:
            dead_paths.append({'boxes': None, 'scores': None, 'allScores': None,
                               'foundAt': None, 'count': None})
            dead_paths[dp_count]['boxes'] = live_paths[olp]['boxes']
            dead_paths[dp_count]['scores'] = live_paths[olp]['scores']
            dead_paths[dp_count]['allScores'] = live_paths[olp]['allScores']
            dead_paths[dp_count]['foundAt'] = live_paths[olp]['foundAt']
            dead_paths[dp_count]['count'] = live_paths[olp]['count']
            dp_count = dp_count + 1

    return sorted_live_paths, dead_paths

def copy_live_to_dead(live_paths, dead_paths, min_len):
    dp_count = len(dead_paths)
    for lp in range(len(live_paths)):
        # path_score = np.mean(live_paths[lp]['scores'])
        # if len(live_paths[lp]['boxes']) >= min_len or path_score > 0.01:
        dead_paths.append({'boxes': None, 'scores': None, 'allScores': None,
                            'foundAt': None, 'count': None})
        dead_paths[dp_count]['boxes'] = live_paths[lp]['boxes']
        dead_paths[dp_count]['scores'] = live_paths[lp]['scores']
        dead_paths[dp_count]['allScores'] = live_paths[lp]['allScores']
        dead_paths[dp_count]['foundAt'] = live_paths[lp]['foundAt']
        dead_paths[dp_count]['count'] = live_paths[lp]['count']
        dp_count = dp_count + 1

    return dead_paths


def score_of_edge(v1, v2, iouth, costtype, avoid_dets, as1, as2, jumpgap):

    N2 = v2['boxes'].shape[0]
    score = np.zeros(N2)
    curent_boxes = v1['boxes'][-1,:]
    tm = min(jumpgap+1, v1['boxes'].shape[0])
    past_boxes = v1['boxes'][-tm, :]
    expected_boxes = curent_boxes + (curent_boxes-past_boxes)/max(1,tm-1)
    ious = bbox_overlaps(expected_boxes, v2['boxes'])
    if ious.any()>1:
        print(ious)
    # pdb.set_trace()
    for i in range(0, N2):
        if ious[i] >= iouth and i not in avoid_dets:
            scores2 = v2['scores'][i]
            if costtype == 'score':
                score[i] = scores2
            elif costtype == 'scoreiou':
                score[i] = (scores2 + ious[i])/2
            elif costtype == 'ioul2':
                score[i] = (scores2 + ious[i])/2
                invl2_diff = 1.0/np.sqrt(np.sum((as1-as2[i,:])**2))
                score[i] += invl2_diff
            elif costtype == 'iou':
                score[i] =  ious[i]
    return score


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

def check_if_sorted(array):
    sorted = True
    for i in range(len(array)-1):
        if array[i]>array[i+1]:
            sorted = False
            break
    return sorted

def are_there_gaps(array):
    gaps = False
    for i in range(len(array)-1):
        if array[i+1] - array[i] > 1 :
            gaps = True
            # print(array[i+1], array[i])
            break
    return gaps


def fill_gaps(paths, min_len_with_gaps=8, minscore=0.3):
    lp_count = len(paths)
    new_paths = []
    filling_gaps = 0
    for lp in range(lp_count):
        path = paths[lp]
        path_score = np.mean(path['scores'])
        if len(path['boxes']) >= min_len_with_gaps or path_score > minscore:
            foundAt = path['foundAt']
            assert sorted(foundAt), 'foundAt should have been sorted i.e., paths should be built incremently'
            if are_there_gaps(foundAt):
                if len(foundAt)<=min_len_with_gaps:
                    continue
                filling_gaps += 1
                numb = foundAt[-1] - foundAt[0] + 1
                new_path = {'boxes': np.zeros((numb,4)), 'scores': np.zeros(numb), 
                            'allScores': np.zeros((numb, path['allScores'].shape[1])),
                            'foundAt': np.zeros(numb, dtype=np.int32)}
                            
                count = 0
                fn = foundAt[0]
                for n in range(len(foundAt)):
                    next_ = foundAt[n]
                    if fn == next_ :
                        new_path['foundAt'][count] =  foundAt[n]
                        new_path['boxes'][count, :]  = path['boxes'][n, :]
                        new_path['scores'][count]  = path['scores'][n]
                        new_path['allScores'][count, :]  = path['allScores'][n, :]
                        count += 1
                        fn += 1
                    else:
                        pfn = fn-1
                        pcount = count -1
                        while fn <= next_:
                            weight = (fn - pfn) / (next_ - pfn)
                            new_path['foundAt'][count] = fn 
                            new_path['boxes'][count,:] = new_path['boxes'][pcount,:] + weight*(path['boxes'][n,:] - new_path['boxes'][pcount,:])
                            new_path['allScores'][count,:] = new_path['allScores'][pcount,:] + weight*(path['allScores'][n,:] - new_path['allScores'][pcount,:])
                            new_path['scores'][count] = new_path['scores'][pcount] + weight*(path['scores'][n] - new_path['scores'][pcount])
                            # print(fn, weight, path['boxes'][n,:] - new_path['boxes'][pcount,:], foundAt)
                            # pdb.set_trace()
                            fn += 1
                            count += 1
                    # pdb.set_trace()
                assert count == numb, 'count {:d} numb {:d} are not equal'.format(count, numb)
            else:
                new_path = {'boxes': path['boxes'], 'scores': path['scores'], 
                            'allScores': path['allScores'],
                            'foundAt': path['foundAt']}
            
            new_paths.append(new_path)
            
            # paths[lp]['labels'] = paths[lp]['labels'][-keep_num:]
    # print('Number of tube paths with gaps are ', filling_gaps)

    return paths
