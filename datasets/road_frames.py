import pandas as pd
import cv2
import torch.utils.data as data
from glob import glob
import numpy as np
from utils.misc import collate_fn
import torch
import random
from PIL import Image
import torch.nn.functional as F
import datasets.video_transforms as T
import json


# class VideoDataset(data.Dataset):

#     def __init__(self, frame_path, video_frame_bbox, frame_keys_list, clip_len, frame_sample_rate,
#                  transforms, crop_size=224, resize_size=256, mode="train", class_num=80):
#         self.video_frame_bbox = video_frame_bbox
#         self.video_frame_list = frame_keys_list
#         self.frame_path = frame_path

#         self.video_frame_list = self.video_frame_list

#         self.crop_size = crop_size
#         self.clip_len = clip_len
#         self.frame_sample_rate = frame_sample_rate
#         self.class_num = class_num
#         self.resize_size = resize_size

#         self.index_cnt = 0
#         self._transforms = transforms
#         self.mode = mode

#         print("rescale size: {}, crop size: {}".format(resize_size, crop_size))

#     def __getitem__(self, index):

#         frame_key = self.video_frame_list[index]
#         print(frame_key)


#         vid, frame_second = frame_key.split(",")
#         timef = int(frame_second) - 900

#         start_img = np.max((timef * 30 - self.clip_len // 2 * self.frame_sample_rate, 0))

#         imgs, target = self.loadvideo(start_img, vid, frame_key)
        
#         if len(target) == 0 or target['boxes'].shape[0] == 0:
#             pass
#         else:
#             if self._transforms is not None:
#                 imgs, target = self._transforms(imgs, target)

#         while len(target) == 0 or target['boxes'].shape[0] == 0:
#             print('resample.')
#             self.index_cnt -= 1
#             index = np.random.randint(len(self.video_frame_list))
#             frame_key = self.video_frame_list[index]
#             vid, frame_second = frame_key.split(",")
#             timef = int(frame_second) - 900

#             start_img = np.max((timef * 30 - self.clip_len // 2 * self.frame_sample_rate, 0))

#             imgs, target = self.loadvideo(start_img, vid, frame_key)

#             if len(target)==0 or target['boxes'].shape[0] == 0:
#                 pass
#             else:
#                 if self._transforms is not None:
#                     imgs, target = self._transforms(imgs, target)

#         imgs = torch.stack(imgs, dim=0)
#         imgs = imgs.permute(1, 0, 2, 3)


#         print(imgs.shape)

#         print(target['image_id'])
#         print(target['boxes'])
#         print(target['raw_boxes'])
#         print(target['labels'])
#         print(target['size'])
#         print(target['orig_size'])
#         print(target['area'])

#         print(rr)
#         return imgs, target

#     def load_annotation(self, sample_id, video_frame_list):

#         num_classes = self.class_num
#         boxes, classes = [], []
#         target = {}

#         first_img = cv2.imread(video_frame_list[0])

#         oh = first_img.shape[0]
#         ow = first_img.shape[1]
#         if oh <= ow:
#             nh = self.resize_size
#             nw = self.resize_size * (ow / oh)
#         else:
#             nw = self.resize_size
#             nh = self.resize_size * (oh / ow)

#         p_t = int(self.clip_len // 2)
#         key_pos = p_t
#         anno_entity = self.video_frame_bbox[sample_id]

#         for i, bbox in enumerate(anno_entity["bboxes"]):
#             label_tmp = np.zeros((num_classes, ))
#             acts_p = anno_entity["acts"][i]
#             for l in acts_p:
#                 label_tmp[l] = 1

#             if np.sum(label_tmp) == 0: continue
#             p_x = np.int(bbox[0] * nw)
#             p_y = np.int(bbox[1] * nh)
#             p_w = np.int(bbox[2] * nw)
#             p_h = np.int(bbox[3] * nh)

#             boxes.append([p_t, p_x, p_y, p_w, p_h])
#             classes.append(label_tmp)

#         boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 5)
#         boxes[:, 1::3].clamp_(min=0, max=int(nw))
#         boxes[:, 2::3].clamp_(min=0, max=nh)

#         if boxes.shape[0]:
#             raw_boxes = F.pad(boxes, (1, 0, 0, 0), value=self.index_cnt)
#         else:
#             raw_boxes = boxes

#         classes = torch.as_tensor(classes, dtype=torch.float32).reshape(-1, num_classes)

#         target["image_id"] = [str(sample_id).replace(",", "_"), key_pos]
#         target['boxes'] = boxes
#         target['raw_boxes'] = raw_boxes
#         target["labels"] = classes
#         target["orig_size"] = torch.as_tensor([int(nh), int(nw)])
#         target["size"] = torch.as_tensor([int(nh), int(nw)])
#         self.index_cnt = self.index_cnt + 1

#         return target

#     def loadvideo(self, start_img, vid, frame_key):
#         video_frame_path = self.frame_path.format(vid)
#         video_frame_list = sorted(glob(video_frame_path + '/*.jpg'))

#         if len(video_frame_list) == 0:
#             print("path doesnt exist", video_frame_path)
#             return [], []
        
#         target = self.load_annotation(frame_key, video_frame_list)

#         start_img = np.max(start_img, 0)
#         end_img = start_img + self.clip_len * self.frame_sample_rate
#         indx_img = list(np.clip(range(start_img, end_img, self.frame_sample_rate), 0, len(video_frame_list) - 1))
#         buffer = []
#         for frame_idx in indx_img:
#             tmp = Image.open(video_frame_list[frame_idx])
#             tmp = tmp.resize((target['orig_size'][1], target['orig_size'][0]))
#             buffer.append(tmp)

#         return buffer, target

#     def __len__(self):
#         return len(self.video_frame_list)


def make_transforms(image_set, cfg):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    print("transform image crop: {}".format(cfg.CONFIG.DATA.IMG_SIZE))
    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSizeCrop_Custom(cfg.CONFIG.DATA.IMG_SIZE),
            T.ColorJitter(),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.Resize_Custom(cfg.CONFIG.DATA.IMG_SIZE),
            normalize,
        ])

    if image_set == 'visual':
        return T.Compose([
            T.Resize_Custom(cfg.CONFIG.DATA.IMG_SIZE),
            normalize,
        ])
    raise ValueError(f'unknown {image_set}')

# def obtain_generated_bboxes_training(input_csv="/xxx/AVA_v2.2/ava_{}_v2.2.csv",
#                                      eval_only=False,
#                                      frame_root="/xxx/frames",
#                                      mode="train"):
#     import os
#     from glob import glob
#     used=[]
#     input_csv = input_csv.format(mode)
#     # frame_root = frame_root.format(mode)

#     video_frame_bbox = {}
#     gt_sheet = pd.read_csv(input_csv, header=None)
#     count = 0
#     frame_keys_list = set()
#     missed_videos = set()

#     for index, row in gt_sheet.iterrows():
#         vid = row[0]
#         if not os.path.isdir(frame_root + "/" + vid + ""):
#             missed_videos.add(vid)
#             continue

#         frame_second = row[1]

#         bbox_conf = row[7]
#         if bbox_conf < 0.8:
#             continue
#         frame_key = "{},{}".format(vid, str(frame_second).zfill(4))

#         frame_keys_list.add(frame_key)

#         count += 1
#         bbox = [row[2], row[3], row[4], row[5]]
#         gt = int(row[6])

#         if frame_key not in video_frame_bbox.keys():
#             video_frame_bbox[frame_key] = {}
#             video_frame_bbox[frame_key]["bboxes"] = [bbox]
#             video_frame_bbox[frame_key]["acts"] = [[gt - 1]]
#         else:
#             if bbox not in video_frame_bbox[frame_key]["bboxes"]:
#                 video_frame_bbox[frame_key]["bboxes"].append(bbox)
#                 video_frame_bbox[frame_key]["acts"].append([gt - 1])
#             else:
#                 idx = video_frame_bbox[frame_key]["bboxes"].index(bbox)
#                 video_frame_bbox[frame_key]["acts"][idx].append(gt - 1)

#     print("missed vids:")
#     print(missed_videos)
#     return video_frame_bbox, list(frame_keys_list)


# def make_image_key(video_id, timestamp):
#     """Returns a unique identifier for a video id & timestamp."""
#     return "%s,%04d" % (video_id, int(timestamp))
















"""

Target is in xmin, ymin, xmax, ymax, label
coordinates are in range of [0, 1] normlised height and width

"""

import json, os
import torch
import pdb, time
import torch.utils as tutils
import pickle
# from .transforms import get_clip_list_resized
import torch.nn.functional as F
import numpy as np
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES =   True
from PIL import Image, ImageDraw
from modules.tube_helper import make_gt_tube
import random as random
from modules import utils 
from random import shuffle

logger = utils.get_logger(__name__)


def make_box_anno(llist):
    box = [llist[2], llist[3], llist[4], llist[5]]
    return [float(b) for b in box]


def read_ava_annotations(anno_file):
    # print(anno_file)
    lines = open(anno_file, 'r').readlines()
    annotations = {}
    is_train = anno_file.find('train') > -1

    cc = 0
    for line in lines:
        cc += 1
        # if cc>500:
        #     break
        line = line.rstrip('\n')
        line_list = line.split(',')
        # print(line_list)
        video_name = line_list[0]
        if video_name not in annotations:
            annotations[video_name] = {}
        time_stamp = float(line_list[1])
        # print(line_list)
        numf = float(line_list[7]) ## or score
        ts = str(int(time_stamp))
        if len(line_list) > 2:
            box = make_box_anno(line_list)
            label = int(line_list[6])
            if ts not in annotations[video_name]:
                annotations[video_name][ts] = [[time_stamp, box, label, numf]]
            else:
                annotations[video_name][ts] += [[time_stamp, box, label, numf]]
        elif not is_train:
            if video_name not in annotations:
                annotations[video_name][ts] = [[time_stamp, None, None, numf]]
            else:
                annotations[video_name][ts] += [[time_stamp, None, None, numf]]

    # for video_name in annotations:
        # print(video_name)
    return annotations



def read_labelmap(labelmap_file):
    """Read label map and class ids."""

    labelmap = {}
    class_ids_map = {}
    name = ""
    class_id = ""
    class_names = []
    print('load label map from ', labelmap_file)
    count = 0
    with open(labelmap_file, "r") as f:
        for line in f:
            # print(line)
            if line.startswith("  name:"):
                name = line.split('"')[1]
            elif line.startswith("  id:") or line.startswith("  label_id:"):
                class_id = int(line.strip().split(" ")[-1])
                labelmap[name] = {'org_id':class_id, 'used_id': count}
                class_ids_map[class_id] = {'used_id':count, 'clsname': name}
                count += 1
                # print(class_id, name)
                class_names.append(name)
    
    # class_names[0]
    print('NUmber of classes are ', count)

    return class_names, class_ids_map, labelmap


def get_box(box, counts):
    box = box.astype(np.float32) - 1
    box[2] += box[0]  #convert width to xmax
    box[3] += box[1]  #converst height to ymax
    for bi in range(4):
        scale = 320 if bi % 2 == 0 else 240
        box[bi] /= scale
        assert 0<=box[bi]<=1.01, box
        # if add_one ==0:
        box[bi] = min(1.0, max(0, box[bi]))
        if counts is None:
            box[bi] = box[bi]*682 if bi % 2 == 0 else box[bi]*512

    return box, counts

def get_frame_level_annos_ucf24(annotations, numf, num_classes, counts=None):
    frame_level_annos = [ {'labeled':True,'ego_label':0,'boxes':[],'labels':[]} for _ in range(numf)]
    add_one = 1
    # if num_classes == 24:
    # add_one = 0
    for tubeid, tube in enumerate(annotations):
    # print('numf00', numf, tube['sf'], tube['ef'])
        for frame_index, frame_num in enumerate(np.arange(tube['sf'], tube['ef'], 1)): # start of the tube to end frame of the tube
            label = tube['label']
            # assert action_id == label, 'Tube label and video label should be same'
            box, counts = get_box(tube['boxes'][frame_index, :].copy(), counts)  # get the box as an array
            frame_level_annos[frame_num]['boxes'].append(box)
            box_labels = np.zeros(num_classes)
            # if add_one == 1:
            box_labels[0] = 1 
            box_labels[label+add_one] = 1
            frame_level_annos[frame_num]['labels'].append(box_labels)
            frame_level_annos[frame_num]['ego_label'] = label+1
            # frame_level_annos[frame_index]['ego_label'][] = 1
            if counts is not None:
                counts[0,0] += 1
                counts[label,1] += 1
        
    return frame_level_annos, counts


def get_frame_level_annos_ava(annotations, numf, num_classes, class_ids_map, counts=None, split='val'):
    frame_level_annos = [ {'labeled':False,'ego_label':-1,'boxes':[],'labels':[]} for _ in range(numf)]
    
    keyframes = []
    skip_count = 0
    timestamps = [ str(i) for i in range(902, 1799)]

    if split == 'train':
        timestamps = [ts for ts in annotations]

    for ts in timestamps:
        boxes = {}
        time_stamp = int(ts)
        frame_num = int((time_stamp - 900) * 30 + 1)
        
        if ts in annotations:
            # pdb.set_trace()
            assert time_stamp == int(annotations[ts][0][0])
            
            for anno in annotations[ts]:
                box_key = '_'.join('{:0.3f}'.format(b) for b in anno[1])
                assert 80>=anno[2]>=1, 'label should be between 1 and 80 but it is {} '.format(anno[2])
                if anno[2] not in class_ids_map:
                    skip_count += 1
                    continue

                class_id = class_ids_map[anno[2]]['used_id']
                # print(class_id)
                if box_key not in boxes:
                    boxes[box_key] = {'box':anno[1], 'labels':np.zeros(num_classes)}

                boxes[box_key]['labels'][class_id+1] = 1
                boxes[box_key]['labels'][0] = 1
                counts[class_id,1] += 1
            
            new_boxes = []
            labels = []
            for box_key in boxes:
                new_boxes.append(boxes[box_key]['box'])
                labels.append(boxes[box_key]['labels'])
            
            if len(new_boxes):
                new_boxes = np.asarray(new_boxes)
                frame_level_annos[frame_num]['boxes'] = new_boxes

                labels = np.asarray(labels)
                frame_level_annos[frame_num]['labels'] = labels

                frame_level_annos[frame_num]['labeled'] = True
                frame_level_annos[frame_num]['ego_label'] = 1


        keyframes.append(frame_num)
        if not frame_level_annos[frame_num]['labeled']:
            frame_level_annos[frame_num]['ego_label'] = 0

    return frame_level_annos, counts, keyframes, skip_count


def get_filtered_tubes_ucf24(annotations):
    filtered_tubes = []
    for tubeid, tube in enumerate(annotations):
        frames = []
        boxes = []
        label = tube['label']
        count = 0
        for frame_index, frame_num in enumerate(np.arange(tube['sf'], tube['ef'], 1)):
            frames.append(frame_num+1)
            box, _ = get_box(tube['boxes'][frame_index, :].copy(), None)
            boxes.append(box)
            count += 1
        assert count == tube['boxes'].shape[0], 'numb: {} count ={}'.format(tube['boxes'].shape[0], count)
        temp_tube = make_gt_tube(frames, boxes, label)
        filtered_tubes.append(temp_tube)
    return filtered_tubes


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


def filter_labels(ids, all_labels, used_labels):
    """Filter the used ids"""
    used_ids = []
    for id in ids:
        label = all_labels[id]
        if label in used_labels:
            used_ids.append(used_labels.index(label))
    
    return used_ids


def get_gt_video_list(anno_file, SUBSETS):
    """Get video list form ground truth videos used in subset 
    and their ground truth tubes """

    with open(anno_file, 'r') as fff:
        final_annots = json.load(fff)

    video_list = []
    for videoname in final_annots['db']:
        if is_part_of_subsets(final_annots['db'][videoname]['split_ids'], SUBSETS):
            video_list.append(videoname)

    return video_list


def get_filtered_tubes(label_key, final_annots, videoname):
    
    key_tubes = final_annots['db'][videoname][label_key]
    all_labels = final_annots['all_'+label_key.replace('tubes','labels')]
    labels = final_annots[label_key.replace('tubes','labels')]
    filtered_tubes = []
    for _ , tube in key_tubes.items():
        label_id = tube['label_id']
        label = all_labels[label_id]
        if label in labels:
            new_label_id = labels.index(label)
            # temp_tube = GtTube(new_label_id)
            frames = []
            boxes = []
            if 'annos' in tube.keys():
                for fn, anno_id in tube['annos'].items():
                    frames.append(int(fn))
                    anno = final_annots['db'][videoname]['frames'][fn]['annos'][anno_id]
                    box = anno['box'].copy()
                    for bi in range(4):
                        assert 0<=box[bi]<=1.01, box
                        box[bi] = min(1.0, max(0, box[bi]))
                        box[bi] = box[bi]*682 if bi % 2 == 0 else box[bi]*512
                    boxes.append(box)
            else:
                for fn in tube['frames']:
                    frames.append(int(fn))

            temp_tube = make_gt_tube(frames, boxes, new_label_id)
            filtered_tubes.append(temp_tube)
            
    return filtered_tubes


def get_filtered_frames(label_key, final_annots, videoname, filtered_gts):
    
    frames = final_annots['db'][videoname]['frames']
    if label_key == 'agent_ness':
        all_labels = []
        labels = []
    else:
        all_labels = final_annots['all_'+label_key+'_labels']
        labels = final_annots[label_key+'_labels']
    
    for frame_id , frame in frames.items():
        frame_name = '{:05d}'.format(int(frame_id))
        if frame['annotated']>0:
            all_boxes = []
            if 'annos' in frame:
                frame_annos = frame['annos']
                for key in frame_annos:
                    anno = frame_annos[key]
                    box = np.asarray(anno['box'].copy())
                    for bi in range(4):
                        assert 0<=box[bi]<=1.01, box
                        box[bi] = min(1.0, max(0, box[bi]))
                        box[bi] = box[bi]*682 if bi % 2 == 0 else box[bi]*512
                    if label_key == 'agent_ness':
                        filtered_ids = [0]
                    else:
                        filtered_ids = filter_labels(anno[label_key+'_ids'], all_labels, labels)

                    if len(filtered_ids)>0:
                        all_boxes.append([box, filtered_ids])
                
            filtered_gts[videoname+frame_name] = all_boxes
            
    return filtered_gts

def get_av_actions(final_annots, videoname):
    label_key = 'av_action'
    frames = final_annots['db'][videoname]['frames']
    all_labels = final_annots['all_'+label_key+'_labels']
    labels = final_annots[label_key+'_labels']
    
    filtered_gts = {}
    for frame_id , frame in frames.items():
        frame_name = '{:05d}'.format(int(frame_id))
        if frame['annotated']>0:
            gts = filter_labels(frame[label_key+'_ids'], all_labels, labels)
            filtered_gts[videoname+frame_name] = gts
            
    return filtered_gts

def get_video_tubes(final_annots, videoname):
    
    tubes = {}
    for key in final_annots['db'][videoname].keys():
        if key.endswith('tubes'):
            filtered_tubes = get_filtered_tubes(key, final_annots, videoname)
            tubes[key] = filtered_tubes
    
    return tubes


def is_part_of_subsets(split_ids, SUBSETS):
    
    is_it = False
    for subset in SUBSETS:
    
        if subset in split_ids:
            is_it = True
    
    return is_it


class VideoDataset(tutils.data.Dataset):
    """
    ROAD Detection dataset class for pytorch dataloader
    """

    def __init__(self, args, train=True, input_type='rgb', transform=None, 
                skip_step=1, full_test=False,crop_size=224, resize_size=256):

        self.num_of_classes = args.CONFIG.DATA.NUM_CLASSES
        self.DATASET = args.CONFIG.DATA.DATASET
        if train == True:
            self.SUBSETS = args.CONFIG.DATA.TRAIN_SUBSETS
        else:
            self.SUBSETS = args.CONFIG.DATA.VAL_SUBSETS

        self.SEQ_LEN = args.CONFIG.DATA.SEQ_LEN
        self.index_cnt = 0
        self.MIN_SEQ_STEP = args.CONFIG.DATA.MIN_SEQ_STEP
        self.MAX_SEQ_STEP = args.CONFIG.DATA.MAX_SEQ_STEP
        # self.MULIT_SCALE = args.MULIT_SCALE
        self.full_test = full_test
        self.skip_step = skip_step #max(skip_step, self.SEQ_LEN*self.MIN_SEQ_STEP/2)
        self.num_steps = max(1, int(self.MAX_SEQ_STEP - self.MIN_SEQ_STEP + 1 )//2)
        # self.input_type = input_type
        self.input_type = input_type+'-images'
        self.train = train
        self.root = args.CONFIG.DATA.DATA_ROOT + args.CONFIG.DATA.DATASET + '/'
        self._imgpath = os.path.join(self.root, self.input_type)
        self.anno_root = self.root
        if len(args.CONFIG.DATA.ANNO_ROOT)>1:
            self.anno_root = args.CONFIG.DATA.ANNO_ROOT 

        self.crop_size = crop_size
        self.resize_size = resize_size


        # self.image_sets = image_sets
        self._transforms = transform
        self.ids = list()
        if self.DATASET == 'road':
            self._make_lists_road()
        elif self.DATASET == 'roadpp':
            self._make_lists_roadpp()
        elif self.DATASET == 'ucf24':
            self._make_lists_ucf24() 
        else:
            raise Exception('Specfiy corect dataset')
        
        self.num_label_types = len(self.label_types)




    def _make_lists_ucf24(self):

        self.anno_file  = os.path.join(self.anno_root, 'pyannot_with_class_names.pkl')

        with open(self.anno_file,'rb') as fff:
            final_annots = pickle.load(fff)
        
        database = final_annots['db']
        self.trainvideos = final_annots['trainvideos']
        ucf_classes = final_annots['classes']
        self.label_types =  ['action_ness', 'action'] #
        # pdb.set_trace()
        self.num_classes_list = [1, 24]
        self.num_classes = 25 # one for action_ness
        
        self.ego_classes = ['Non_action']  +  ucf_classes
        self.num_ego_classes = len(self.ego_classes)
        
        counts = np.zeros((24, 2), dtype=np.int32)
    
        ratios = [1.0, 1.1, 1.1, 0.9, 1.1, 0.8, 0.7, 0.8, 1.1, 1.4, 1.0, 0.8, 0.7, 1.2, 1.0, 0.8, 0.7, 1.2, 1.2, 1.0, 0.9]
    
        self.video_list = []
        self.numf_list = []
        
        frame_level_list = []

        default_ego_label = np.zeros(self.num_ego_classes)
        default_ego_label[0] = 1
        total_labeled_frame = 0
        total_num_frames = 0

        for videoname in sorted(database.keys()):    
            is_part = 1
            if 'train' in self.SUBSETS and videoname not in self.trainvideos:
                continue
            elif 'test' in self.SUBSETS and videoname in self.trainvideos:
                continue
            # print(database[videoname].keys())
            action_id = database[videoname]['label']
            annotations = database[videoname]['annotations']
            
            numf = database[videoname]['numf']
            self.numf_list.append(numf)
            self.video_list.append(videoname)
            
            # frames = database[videoname]['frames']
            
            frame_level_annos, counts = get_frame_level_annos_ucf24(annotations, numf, self.num_classes, counts)

            frames_with_boxes = 0
            for frame_index in range(numf): #frame_level_annos:
                if len(frame_level_annos[frame_index]['labels'])>0:
                    frames_with_boxes += 1
                frame_level_annos[frame_index]['labels'] = np.asarray(frame_level_annos[frame_index]['labels'], dtype=np.float32)
                frame_level_annos[frame_index]['boxes'] = np.asarray(frame_level_annos[frame_index]['boxes'], dtype=np.float32)

            total_labeled_frame += frames_with_boxes
            total_num_frames += numf

            # logger.info('Frames with Boxes are {:d} out of {:d} in {:s}'.format(frames_with_boxes, numf, videoname))
            frame_level_list.append(frame_level_annos)  
            ## make ids
            start_frames = [ f for f in range(numf-self.MIN_SEQ_STEP*self.SEQ_LEN, -1,  -self.skip_step)]
            
            if self.full_test and 0 not in start_frames:
                start_frames.append(0)
            # logger.info('number of start frames: '+ str(len(start_frames)))
            for frame_num in start_frames:
                step_list = [s for s in range(self.MIN_SEQ_STEP, self.MAX_SEQ_STEP+1) if numf-s*self.SEQ_LEN>=frame_num]
                shuffle(step_list)
                # print(len(step_list), self.num_steps)
                for s in range(min(self.num_steps, len(step_list))):
                    video_id = self.video_list.index(videoname)
                    self.ids.append([video_id, frame_num ,step_list[s]])

        logger.info('Labeled frames {:d}/{:d}'.format(total_labeled_frame, total_num_frames))
        # pdb.set_trace()
        ptrstr = '\n'
        self.frame_level_list = frame_level_list
        self.all_classes = [['action_ness'], ucf_classes.copy()]
        for k, name in enumerate(self.label_types):
            labels = self.all_classes[k]
            # self.num_classes_list.append(len(labels))
            for c, cls_ in enumerate(labels): # just to see the distribution of train and test sets
                ptrstr += '-'.join(self.SUBSETS) + ' {:05d} label: ind={:02d} name:{:s}\n'.format(
                                                counts[c,k] , c, cls_)
        
        ptrstr += 'Number of ids are {:d}\n'.format(len(self.ids))
        ptrstr += 'Labeled frames {:d}/{:d}'.format(total_labeled_frame, total_num_frames)
        self.childs = {}
        self.num_videos = len(self.video_list)
        self.print_str = ptrstr
        
    
    def _make_lists_roadpp(self):
        
        # if self.MODE =='train':
        #     self.anno_file  = os.path.join(self.root, 'road_plus_plus_trainval_v1.0.json')
        # else:
        #     self.anno_file  = os.path.join(self.root, 'road_plus_plus_test_v1.0.json')
        
        self.anno_file  = os.path.join(self.root, 'road_plus_plus_trainval_v1.0.json')
        with open(self.anno_file,'r') as fff:
            final_annots = json.load(fff)
        
        database = final_annots['db']
        
        # self.label_types =  final_annots['label_types'] #['agent', 'action', 'loc', 'duplex', 'triplet'] #
        self.label_types = ['agent', 'action', 'loc'] #
        # print(self.label_types)
        # print(rr)

        num_label_type = len(self.label_types)
        self.num_classes = 1 ## one for presence
        self.num_classes_list = [1]
        for name in self.label_types: 
            logger.info('Number of {:s}: all :: {:d} to use: {:d}'.format(name, 
                len(final_annots['all_'+name+'_labels']),len(final_annots[name+'_labels'])))
            numc = len(final_annots[name+'_labels'])
            self.num_classes_list.append(numc)
            self.num_classes += numc
        
        self.ego_classes = final_annots['av_action_labels']
        self.num_ego_classes = len(self.ego_classes)

        # counts = np.zeros((len(final_annots[self.label_types[-1] + '_labels']), num_label_type), dtype=np.int32)
        counts = np.zeros((len(final_annots[self.label_types[0] + '_labels']) + len(final_annots[self.label_types[1] + '_labels']) +len(final_annots[self.label_types[2] + '_labels'])  , num_label_type), dtype=np.int32)


        self.video_list = []
        self.numf_list = []
        frame_level_list = []

        for videoname in sorted(database.keys()):
            # print(is_part_of_subsets(final_annots['db'][videoname]['split_ids'], self.SUBSETS))
            if not is_part_of_subsets(final_annots['db'][videoname]['split_ids'], self.SUBSETS):
                continue
            
            numf = database[videoname]['numf']
            self.numf_list.append(numf)
            self.video_list.append(videoname)
            
            frames = database[videoname]['frames']
            # print(numf)
            frame_level_annos = [ {'labeled':False,'ego_label':-1,'boxes':np.asarray([]),'labels':np.asarray([])} for _ in range(numf)]

            frame_nums = [int(f) for f in frames.keys()]
            frames_with_boxes = 0
            for frame_num in sorted(frame_nums): #loop from start to last possible frame which can make a legit sequence
                frame_id = str(frame_num)
                if frame_id in frames.keys() and frames[frame_id]['annotated']>0:
                    
                    frame_index = frame_num-1  
                    frame_level_annos[frame_index]['labeled'] = True 
                    # frame_level_annos[frame_index]['ego_label'] = frames[frame_id]['av_action_ids'][0]
                    
                    frame = frames[frame_id]
                    if 'annos' not in frame.keys():
                        frame = {'annos':{}}
                    
                    all_boxes = []
                    all_labels = []
                    frame_annos = frame['annos']
                    # temp_img = cv2.imread('../roadpp/rgb-images/'+videoname+'/{:05d}.jpg'.format(frame_num))
                    for key in frame_annos:
                        width, height = frame['width'], frame['height']
                        anno = frame_annos[key]
                        box = anno['box']
                        
                        assert box[0]<box[2] and box[1]<box[3], box
                        assert width==1920 and height==1280, (width, height, box)
                        
                        # temp_img = cv2.rectangle(temp_img, (int(box[0]*1920),int(box[1]*1280)), (int(box[2]*1920),int(box[3]*1280)), (255,0,0), 2)
                        # cv2.imwrite('temp_img.png',temp_img)
                        for bi in range(4):
                            assert 0<=box[bi]<=1.01, box
                            box[bi] = min(1.0, max(0, box[bi]))
                        
                        all_boxes.append(box)
                        box_labels = np.zeros(self.num_classes-1)
                        list_box_labels = []
                        cc = 1
                        for idx, name in enumerate(self.label_types):
                            # print(idx,name)
                            filtered_ids = filter_labels(anno[name+'_ids'], final_annots['all_'+name+'_labels'], final_annots[name+'_labels'])
                            list_box_labels.append(filtered_ids)
                            for fid in filtered_ids:
                                box_labels[fid+cc-1] = 1
                                # box_labels[0] = 1
                            cc += self.num_classes_list[idx+1]

                        all_labels.append(box_labels)

                        # for box_labels in all_labels:
                        for k, bls in enumerate(list_box_labels):
                            for l in bls:
                                counts[l, k] += 1 
                    # print(videoname,frame_num)
                    # print(rr)
                    all_labels = np.asarray(all_labels, dtype=np.float32)
                    all_boxes = np.asarray(all_boxes, dtype=np.float32)

                    if all_boxes.shape[0]>0:
                        frames_with_boxes += 1    
                    frame_level_annos[frame_index]['labels'] = all_labels
                    frame_level_annos[frame_index]['boxes'] = all_boxes

            logger.info('Frames with Boxes are {:d} out of {:d} in {:s}'.format(frames_with_boxes, numf, videoname))
            frame_level_list.append(frame_level_annos)  

            ## make ids
            start_frames = [ f for f in range(numf-self.MIN_SEQ_STEP*self.SEQ_LEN, 1,  -self.skip_step)]
            if self.full_test and 1 not in start_frames:
                start_frames.append(1)
            logger.info('number of start frames: '+ str(len(start_frames)))
            for frame_num in start_frames:
                step_list = [s for s in range(self.MIN_SEQ_STEP, self.MAX_SEQ_STEP+1) if numf-s*self.SEQ_LEN>=frame_num]
                shuffle(step_list)
                # print(len(step_list), self.num_steps)
                for s in range(min(self.num_steps, len(step_list))):
                    video_id = self.video_list.index(videoname)
                    if len(frame_level_list[video_id][frame_num+int(self.SEQ_LEN/2)]['boxes']) >0:
                            self.ids.append([video_id, frame_num ,step_list[s]])

        # pdb.set_trace()
        ptrstr = ''
        self.frame_level_list = frame_level_list
        self.all_classes = [['agent_ness']]
        for k, name in enumerate(self.label_types):
            labels = final_annots[name+'_labels']
            self.all_classes.append(labels)
            # self.num_classes_list.append(len(labels))
            for c, cls_ in enumerate(labels): # just to see the distribution of train and test sets
                ptrstr += '-'.join(self.SUBSETS) + ' {:05d} label: ind={:02d} name:{:s}\n'.format(
                                                counts[c,k] , c, cls_)
        
        ptrstr += 'Number of ids are {:d}\n'.format(len(self.ids))

        self.label_types = ['agent_ness'] + self.label_types
        self.childs = {'duplex_childs':final_annots['duplex_childs'], 'triplet_childs':final_annots['triplet_childs']}
        self.num_videos = len(self.video_list)
        self.print_str = ptrstr



    def _make_lists_road(self):
        
        self.anno_file  = os.path.join(self.root, 'road_trainval_v1.0.json')

        with open(self.anno_file,'r') as fff:
            final_annots = json.load(fff)
        
        database = final_annots['db']
        
        # self.label_types =  final_annots['label_types'] #['agent', 'action', 'loc', 'duplex', 'triplet'] #
        self.label_types = ['agent', 'action', 'loc']
        num_label_type = len(self.label_types)
        self.num_classes = 1 ## one for presence
        self.num_classes_list = [1]
        for name in self.label_types: 
            logger.info('Number of {:s}: all :: {:d} to use: {:d}'.format(name, 
                len(final_annots['all_'+name+'_labels']),len(final_annots[name+'_labels'])))
            numc = len(final_annots[name+'_labels'])
            self.num_classes_list.append(numc)
            self.num_classes += numc
        
        self.ego_classes = final_annots['av_action_labels']
        self.num_ego_classes = len(self.ego_classes)
        
        counts = np.zeros(((len(final_annots[self.label_types[0] + '_labels'])+len(final_annots[self.label_types[1] + '_labels'])+len(final_annots[self.label_types[2] + '_labels'])), num_label_type), dtype=np.int32)

        self.video_list = []
        self.numf_list = []
        frame_level_list = []

        for videoname in sorted(database.keys()):

            if not is_part_of_subsets(final_annots['db'][videoname]['split_ids'], self.SUBSETS):
                continue

            numf = database[videoname]['numf']
            self.numf_list.append(numf)
            self.video_list.append(videoname)
            
            frames = database[videoname]['frames']
            frame_level_annos = [ {'labeled':False,'ego_label':-1,'boxes':np.asarray([]),'labels':np.asarray([])} for _ in range(numf)]

            frame_nums = [int(f) for f in frames.keys()]
            frames_with_boxes = 0
            for frame_num in sorted(frame_nums): #loop from start to last possible frame which can make a legit sequence
                frame_id = str(frame_num)
                if frame_id in frames.keys() and frames[frame_id]['annotated']>0:
                    
                    frame_index = frame_num-1  
                    frame_level_annos[frame_index]['labeled'] = True 
                    frame_level_annos[frame_index]['ego_label'] = frames[frame_id]['av_action_ids'][0]
                    
                    frame = frames[frame_id]
                    if 'annos' not in frame.keys():
                        frame = {'annos':{}}
                    
                    all_boxes = []
                    all_labels = []
                    frame_annos = frame['annos']
                    for key in frame_annos:
                        width, height = frame['width'], frame['height']
                        anno = frame_annos[key]
                        box = anno['box']
                        
                        assert box[0]<box[2] and box[1]<box[3], box
                        assert width==1280 and height==960, (width, height, box)

                        for bi in range(4):
                            assert 0<=box[bi]<=1.01, box
                            box[bi] = min(1.0, max(0, box[bi]))
                        
                        all_boxes.append(box)
                        box_labels = np.zeros(self.num_classes-1)
                        list_box_labels = []
                        cc = 1
                        for idx, name in enumerate(self.label_types):
                            filtered_ids = filter_labels(anno[name+'_ids'], final_annots['all_'+name+'_labels'], final_annots[name+'_labels'])
                            list_box_labels.append(filtered_ids)
                            for fid in filtered_ids:
                                box_labels[fid+cc-1] = 1
                                # box_labels[0] = 1
                            cc += self.num_classes_list[idx+1]

                        all_labels.append(box_labels)

                        # for box_labels in all_labels:
                        for k, bls in enumerate(list_box_labels):
                            for l in bls:
                                counts[l, k] += 1 

                    all_labels = np.asarray(all_labels, dtype=np.float32)
                    all_boxes = np.asarray(all_boxes, dtype=np.float32)

                    if all_boxes.shape[0]>0:
                        frames_with_boxes += 1    
                    frame_level_annos[frame_index]['labels'] = all_labels
                    frame_level_annos[frame_index]['boxes'] = all_boxes

            logger.info('Frames with Boxes are {:d} out of {:d} in {:s}'.format(frames_with_boxes, numf, videoname))
            frame_level_list.append(frame_level_annos)  

            ## make ids
            start_frames = [ f for f in range(numf-self.MIN_SEQ_STEP*self.SEQ_LEN, 1,  -self.skip_step)]
            # if self.full_test and 0 not in start_frames:
            #     start_frames.append(0)
            logger.info('number of start frames: '+ str(len(start_frames)))
            for frame_num in start_frames:
                step_list = [s for s in range(self.MIN_SEQ_STEP, self.MAX_SEQ_STEP+1) if numf-s*self.SEQ_LEN>=frame_num]
                shuffle(step_list)
                # print(len(step_list), self.num_steps)
                for s in range(min(self.num_steps, len(step_list))):
                    video_id = self.video_list.index(videoname)
                    if len(frame_level_list[video_id][frame_num+int(self.SEQ_LEN/2)]['boxes']) >0:
                        self.ids.append([video_id, frame_num ,step_list[s]])
        # print(rr) 
        # pdb.set_trace()
        ptrstr = ''
        self.frame_level_list = frame_level_list
        self.all_classes = [['agent_ness']]
        for k, name in enumerate(self.label_types):
            
            labels = final_annots[name+'_labels']
            self.all_classes.append(labels)

            # self.num_classes_list.append(len(labels))
            for c, cls_ in enumerate(labels): # just to see the distribution of train and test sets
                ptrstr += '-'.join(self.SUBSETS) + ' {:05d} label: ind={:02d} name:{:s}\n'.format(
                                                counts[c,k] , c, cls_)
        
        ptrstr += 'Number of ids are {:d}\n'.format(len(self.ids))

        self.label_types = ['agent_ness'] + self.label_types
        self.childs = {'duplex_childs':final_annots['duplex_childs'], 'triplet_childs':final_annots['triplet_childs']}
        self.num_videos = len(self.video_list)
        self.print_str = ptrstr
        
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        id_info = self.ids[index]
        
        video_id, start_frame, step_size = id_info
        videoname = self.video_list[video_id]
        images = []
        frame_num = start_frame
        ego_labels = np.zeros(self.SEQ_LEN)-1
        all_boxes = []
        labels = []
        ego_labels = []
        mask = np.zeros(self.SEQ_LEN, dtype=np.int)
        indexs = []
        target = {}

        first_img = cv2.imread(self._imgpath + '/{:s}/{:05d}.jpg'.format(videoname, frame_num+int(self.SEQ_LEN/2)))

        oh = first_img.shape[0]
        ow = first_img.shape[1]
        if oh <= ow:
            nh = self.resize_size
            nw = self.resize_size * (ow / oh)
        else:
            nw = self.resize_size
            nh = self.resize_size * (oh / ow)


        p_t = int(self.SEQ_LEN // 2)
        key_pos = p_t
        target["image_id"] = [videoname+"_"+str(frame_num+key_pos), key_pos]
        target["orig_size"] = torch.as_tensor([int(nh), int(nw)])
        target["size"] = torch.as_tensor([int(nh), int(nw)])
        

        for i in range(self.SEQ_LEN):
            indexs.append(frame_num)
            img_name = self._imgpath + '/{:s}/{:05d}.jpg'.format(videoname, frame_num)
            # img_name = self._imgpath + '/{:s}/img_{:05d}.jpg'.format(videoname, frame_num)
            img = Image.open(img_name)
            img = img.resize((target['orig_size'][1], target['orig_size'][0]))
            images.append(img)
            if self.frame_level_list[video_id][frame_num]['labeled']:
                mask[i] = 1
                all_boxes.append(self.frame_level_list[video_id][frame_num]['boxes'].copy())
                labels.append(self.frame_level_list[video_id][frame_num]['labels'].copy())
                # ego_labels.append(self.frame_level_list[video_id][frame_num]['ego_label'])
            else:
                self.index_cnt -= 1
                all_boxes.append(np.asarray([]))
                labels.append(np.asarray([]))
                # ego_labels.append(-1)            
            frame_num += step_size
        
        imgs, target = self._transforms(images, target)


        imgs = torch.stack(imgs, dim=0)
        imgs = imgs.permute(1, 0, 2, 3)


        keyframe_box = all_boxes[key_pos]
        keyframe_label = labels[key_pos]

        boxes = []
        for i, bbox in enumerate(keyframe_box):
            boxes.append([p_t, bbox[0],bbox[1],bbox[2],bbox[3]])

        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 5)
        boxes[:, 1::3].clamp_(min=0, max=int(nw))
        boxes[:, 2::3].clamp_(min=0, max=nh)

        if boxes.shape[0]:
            raw_boxes = F.pad(boxes, (1, 0, 0, 0), value=self.index_cnt)
        else:
            raw_boxes = boxes
       
        for i, bbox in enumerate(raw_boxes):
            raw_boxes[i][2] = np.int(raw_boxes[i][2] * nw)
            raw_boxes[i][3] = np.int(raw_boxes[i][3] * nh)
            raw_boxes[i][4] = np.int(raw_boxes[i][4] * nw)
            raw_boxes[i][5] = np.int(raw_boxes[i][5] * nh)



        classes = torch.as_tensor(keyframe_label, dtype=torch.float32).reshape(-1, self.num_of_classes)

        target['boxes'] = boxes
        target['raw_boxes'] = raw_boxes
        target["labels"] = classes
        self.index_cnt = self.index_cnt + 1

        # print('img',imgs.shape)
        # print('tar',target)
        # print('tar shape',target.shape)
        # print(rr)

        return imgs, target


def build_dataloader(cfg):


    train_dataset = VideoDataset(cfg, train=True, skip_step=cfg.CONFIG.DATA.train_skip_step, transform=make_transforms("train", cfg),resize_size=cfg.CONFIG.DATA.IMG_RESHAPE_SIZE,crop_size=cfg.CONFIG.DATA.IMG_SIZE)


    val_dataset = VideoDataset(cfg, train=False, transform=make_transforms("val", cfg), skip_step=cfg.CONFIG.DATA.skip_step, full_test=True,resize_size=cfg.CONFIG.DATA.IMG_SIZE,crop_size=cfg.CONFIG.DATA.IMG_SIZE)

    # train_bbox_json = json.load(open(cfg.CONFIG.DATA.ANNO_PATH.format("train")))
    # train_video_frame_bbox, train_frame_keys_list = train_bbox_json["video_frame_bbox"], train_bbox_json["frame_keys_list"]

    # train_dataset = VideoDataset(cfg.CONFIG.DATA.DATA_PATH,
    #                              train_video_frame_bbox,
    #                              train_frame_keys_list,
    #                              transforms=make_transforms("train", cfg),
    #                              frame_sample_rate=cfg.CONFIG.DATA.FRAME_RATE,
    #                              clip_len=cfg.CONFIG.DATA.TEMP_LEN,
    #                              resize_size=cfg.CONFIG.DATA.IMG_RESHAPE_SIZE,
    #                              crop_size=cfg.CONFIG.DATA.IMG_SIZE,
    #                              mode="train")

    # val_bbox_json = json.load(open(cfg.CONFIG.DATA.ANNO_PATH.format("val")))
    # val_video_frame_bbox, val_frame_keys_list = val_bbox_json["video_frame_bbox"], val_bbox_json["frame_keys_list"]

    # val_dataset = VideoDataset(cfg.CONFIG.DATA.DATA_PATH,
    #                            val_video_frame_bbox,
    #                            val_frame_keys_list,
    #                            transforms=make_transforms("val", cfg),
    #                            frame_sample_rate=cfg.CONFIG.DATA.FRAME_RATE,
    #                            clip_len=cfg.CONFIG.DATA.TEMP_LEN,
    #                            resize_size=cfg.CONFIG.DATA.IMG_SIZE,
    #                            crop_size=cfg.CONFIG.DATA.IMG_SIZE,
    #                            mode="val")

    if cfg.DDP_CONFIG.DISTRIBUTED:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
        batch_sampler_train = torch.utils.data.BatchSampler(train_sampler, cfg.CONFIG.TRAIN.BATCH_SIZE, drop_last=True)
    else:
        train_sampler = None
        val_sampler = None
        batch_sampler_train = None

    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=(train_sampler is None),
                                               num_workers=9, pin_memory=True, batch_sampler=batch_sampler_train,
                                               collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=cfg.CONFIG.VAL.BATCH_SIZE, shuffle=(val_sampler is None),
        num_workers=9, sampler=val_sampler, pin_memory=True, collate_fn=collate_fn)

    # print(cfg.CONFIG.DATA.ANNO_PATH.format("train"), cfg.CONFIG.DATA.ANNO_PATH.format("val"))

    return train_loader, val_loader, train_sampler, val_sampler, None

def reverse_norm(imgs):
    img = imgs
    mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    img = (img * std + mean) * 255.0
    img = img.transpose((1, 2, 0))[..., ::-1].astype(np.uint8)
    return img








