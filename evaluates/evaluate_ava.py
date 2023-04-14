import json

import torch

from utils.utils import read_labelmap
from evaluates.utils import object_detection_evaluation, standard_fields
import numpy as np
import time
from utils.box_ops import box_iou
import torch
import math


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class STDetectionEvaluater(object):
    '''
    evaluater class designed for multi-iou thresholds
        based on https://github.com/activitynet/ActivityNet/blob/master/Evaluation/get_ava_performance.py
    parameters:
        dataset that provide GT annos, in the format of AWSCVMotionDataset
        tiou_thresholds: a list of iou thresholds
    attributes:
        clear(): clear detection results, GT is kept
        load_detection_from_path(), load anno from a list of path, in the format of [confi x1 y1 x2 y2 scoresx15]
        evaluate(): run evaluation code
    '''

    def __init__(self, label_path, tiou_thresholds=[0.5], load_from_dataset=False, class_num=60):
        self.label_path = label_path
        # print('lab_path', self.label_path)
        categories, class_whitelist = read_labelmap(self.label_path)
        # print('categories', categories)
        # print('class_whitelist', class_whitelist)
        
        self.class_num = class_num
        # print('self.class_num', self.class_num)
        

        if class_num == 80:
            self.exclude_keys = []
            f = open("datasets/assets/ava_val_excluded_timestamps_v2.1.csv")
            while True:
                line = f.readline().strip()
                if not line: break
                self.exclude_keys.append(line.replace(",", "_"))
            f.close()
        else:
            self.exclude_keys = []
        print("exclude keys:", self.exclude_keys)
        self.categories = categories
        self.tiou_thresholds = tiou_thresholds
        self.lst_pascal_evaluator = []
        self.load_from_dataset = load_from_dataset
        self.class_whitelist = class_whitelist
        for iou in self.tiou_thresholds:
            self.lst_pascal_evaluator.append(
                object_detection_evaluation.PascalDetectionEvaluator(categories, matching_iou_threshold=iou))

    def clear(self):
        for evaluator in self.lst_pascal_evaluator:
            evaluator.clear()

    def load_GT_from_path(self, file_lst):
        # loading data from files
        t_end = time.time()
        sample_dict_per_image = {}
        for path in file_lst:
            data = open(path).readlines()
            for line in data:
                image_key = line.split(' [')[0]
                if image_key in self.exclude_keys:
                    print(image_key, "excluded")
                    continue
                data = line.split(' [')[1].split(']')[0].split(',')
                data = [float(x) for x in data]
                scores = np.array(data[6:])
                if not image_key in sample_dict_per_image:
                    sample_dict_per_image[image_key] = {
                        'bbox': [],
                        'labels': [],
                        'scores': [],
                    }
                for x in range(len(scores)):
                    if scores[x] <= 1e-2: continue
                    if self.class_num != 80 or x + 1 in self.class_whitelist:
                        sample_dict_per_image[image_key]['bbox'].append(
                            np.asarray([data[2], data[3], data[4], data[5]], dtype=float)
                        )
                        sample_dict_per_image[image_key]['labels'].append(x + 1)
                        sample_dict_per_image[image_key]['scores'].append(scores[x])
        # write into evaluator
        for image_key, info in sample_dict_per_image.items():
            if len(info['bbox']) == 0: continue
            for evaluator in self.lst_pascal_evaluator:
                evaluator.add_single_ground_truth_image_info(
                    image_key, {
                        standard_fields.InputDataFields.groundtruth_boxes:
                            np.vstack(info['bbox']),
                        standard_fields.InputDataFields.groundtruth_classes:
                            np.array(info['labels'], dtype=int),
                        standard_fields.InputDataFields.groundtruth_difficult:
                            np.zeros(len(info['bbox']), dtype=bool)
                    })
        print("STDetectionEvaluater: test GT loaded in {:.3f}s".format(time.time() - t_end))

    def load_detection_from_path(self, file_lst):
        # loading data from files
        t_end = time.time()
        sample_dict_per_image = {}

        n = 0
        for path in file_lst:
            print("loading ", path)
            data = open(path).readlines()
            for line in data:
                image_key = line.split(' [')[0]
                if image_key in self.exclude_keys:
                    print("excluded", image_key)
                    continue
                data = line.split(' [')[1].split(']')[0].split(',')
                data = [float(x) for x in data]

                scores = np.array(data[4:self.class_num + 4])

                if not image_key in sample_dict_per_image:
                    sample_dict_per_image[image_key] = {
                        'bbox': [],
                        'labels': [],
                        'scores': [],
                    }
                # if image_key=='1j20qq1JyX4_16244':
                #     n+=1

                for x in range(len(scores)):
                    # if scores[x] <= 1e-2: continue
                    if self.class_num != 80 or x + 1 in self.class_whitelist:
                        sample_dict_per_image[image_key]['bbox'].append(
                            np.asarray([data[0], data[1], data[2], data[3]], dtype=float)
                        )
                        sample_dict_per_image[image_key]['labels'].append(x+1)
                        sample_dict_per_image[image_key]['scores'].append(scores[x])
        print("start adding into evaluator")
        count = 0
        for image_key, info in sample_dict_per_image.items():
            if count % 500 == 0:
                print(count, len(sample_dict_per_image.keys()))
            if len(info['bbox']) == 0:
                print(count)
                continue
            #sorted by confidence:
            boxes, labels, scores = np.vstack(info['bbox']), np.array(info['labels'], dtype=int), np.array(info['scores'], dtype=float)
            index = np.argsort(-scores)
            #print('scores',scores[index])
            #exit()
            for evaluator in self.lst_pascal_evaluator:
                evaluator.add_single_detected_image_info(
                    image_key, {
                        standard_fields.DetectionResultFields.detection_boxes:
                            boxes[index],
                        standard_fields.DetectionResultFields.detection_classes:
                            labels[index],
                        standard_fields.DetectionResultFields.detection_scores:
                            scores[index]
                    })
            count += 1


    def evaluate(self):
        result = {}
        mAP = []
        for x, iou in enumerate(self.tiou_thresholds):
            evaluator = self.lst_pascal_evaluator[x]
            metrics = evaluator.evaluate()
            result.update(metrics)
            mAP.append(metrics['PascalBoxes_Precision/mAP@{}IOU'.format(iou)])
        return mAP, result

class STDetectionEvaluaterSinglePerson(object):
    '''
    evaluater class designed for multi-iou thresholds
        based on https://github.com/activitynet/ActivityNet/blob/master/Evaluation/get_ava_performance.py
    parameters:
        dataset that provide GT annos, in the format of AWSCVMotionDataset
        tiou_thresholds: a list of iou thresholds
    attributes:
        clear(): clear detection results, GT is kept
        load_detection_from_path(), load anno from a list of path, in the format of [confi x1 y1 x2 y2 scoresx15]
        evaluate(): run evaluation code
    '''

    def __init__(self, label_path, tiou_thresholds=[0.5], load_from_dataset=False,
                 threshold_size_min=0 * 0, threshold_size_max=555 * 555):
        self.label_path = label_path
        categories, class_whitelist = read_labelmap(self.label_path)
        self.categories = categories
        self.tiou_thresholds = tiou_thresholds
        self.lst_pascal_evaluator = []
        self.load_from_dataset=load_from_dataset
        for iou in self.tiou_thresholds:
            self.lst_pascal_evaluator.append(
                object_detection_evaluation.PascalDetectionEvaluator(categories[:1], matching_iou_threshold=iou))

        self.threshold_size_min = threshold_size_min
        self.threshold_size_max = threshold_size_max

    def clear(self):
        for evaluator in self.lst_pascal_evaluator:
            evaluator.clear()

    def load_GT_from_path(self, file_lst):
        # loading data from files
        t_end = time.time()
        sample_dict_per_image = {}

        bbox_count = 0
        for path in file_lst:
            data = open(path).readlines()
            for line in data:
                image_key = line.split(' [')[0]
                data = line.split(' [')[1].split(']')[0].split(',')
                data = [float(x) for x in data]
                scores = np.array([1])
                x1, y1, x2, y2 = data[2], data[3], data[4], data[5]
                if (x2 - x1) * (y2 - y1) < self.threshold_size_min or (x2 - x1) * (y2 - y1) > self.threshold_size_max:
                    continue
                bbox_count += 1
                if not image_key in sample_dict_per_image:
                    sample_dict_per_image[image_key] = {
                        'bbox': [],
                        'labels': [],
                        'scores': [],
                    }
                for x in range(len(scores)):
                    if scores[x] <= 1e-2: continue
                    sample_dict_per_image[image_key]['bbox'].append(
                        np.asarray([data[2], data[3], data[4], data[5]], dtype=float)
                    )
                    sample_dict_per_image[image_key]['labels'].append(x + 1)
                    sample_dict_per_image[image_key]['scores'].append(scores[x])
        print("gt: min: {}, max: {}".format(self.threshold_size_min, self.threshold_size_max), bbox_count)
        # print(sample_dict_per_image['u1ltv6r14KQ_5875'])
        # write into evaluator
        for image_key, info in sample_dict_per_image.items():
            if len(info['bbox']) == 0: continue
            for evaluator in self.lst_pascal_evaluator:
                evaluator.add_single_ground_truth_image_info(
                    image_key, {
                        standard_fields.InputDataFields.groundtruth_boxes:
                            np.vstack(info['bbox']),
                        standard_fields.InputDataFields.groundtruth_classes:
                            np.array(info['labels'], dtype=int),
                        standard_fields.InputDataFields.groundtruth_difficult:
                            np.zeros(len(info['bbox']), dtype=bool)
                    })
        print("STDetectionEvaluater: test GT loaded in {:.3f}s".format(time.time() - t_end))

    def load_detection_from_path(self, file_lst):
        # loading data from files
        t_end = time.time()
        sample_dict_per_image = {}

        n = 0
        for path in file_lst:
            print("loading ", path)
            data = open(path).readlines()
            for line in data:
                image_key = line.split(' [')[0]
                data = line.split(' [')[1].split(']')[0].split(',')
                data = [float(x) for x in data]
                if data[-1] > 0:
                    scores = np.array(data[-1:])
                else:
                    continue
                # scores = np.array(data[4:]).max(keepdims=True)
                # scores = np.array(data[4:])
                x1, y1, x2, y2 = data[0], data[1], data[2], data[3]
                if (x2 - x1) * (y2 - y1) < self.threshold_size_min or (x2 - x1) * (y2 - y1) > self.threshold_size_max:
                    continue

                if not image_key in sample_dict_per_image:
                    sample_dict_per_image[image_key] = {
                        'bbox': [],
                        'labels': [],
                        'scores': [],
                    }
                if image_key=='1j20qq1JyX4_16244':
                    n+=1

                for x in range(len(scores)):
                    # if scores[x] <= 1e-1: continue
                    sample_dict_per_image[image_key]['bbox'].append(
                        np.asarray([data[0], data[1], data[2], data[3]], dtype=float)
                    )
                    sample_dict_per_image[image_key]['labels'].append(x+1)
                    sample_dict_per_image[image_key]['scores'].append(scores[x])
        print("start adding into evaluator")
        count = 0
        for image_key, info in sample_dict_per_image.items():
            if count % 500 == 0:
                print(count, len(sample_dict_per_image.keys()))
            if len(info['bbox']) == 0:
                print(count)
                continue
            #sorted by confidence:
            boxes, labels, scores = np.vstack(info['bbox']), np.array(info['labels'], dtype=int), np.array(info['scores'], dtype=float)
            # scores = np.ones((len(boxes), ))
            index = np.argsort(-scores)
            #print('scores',scores[index])
            #exit()
            for evaluator in self.lst_pascal_evaluator:
                evaluator.add_single_detected_image_info(
                    image_key, {
                        standard_fields.DetectionResultFields.detection_boxes:
                            boxes[index],
                        standard_fields.DetectionResultFields.detection_classes:
                            labels[index],
                        standard_fields.DetectionResultFields.detection_scores:
                            scores[index]
                    })
            count += 1


    def evaluate(self):
        result = {}
        mAP = []
        for x, iou in enumerate(self.tiou_thresholds):
            evaluator = self.lst_pascal_evaluator[x]
            metrics = evaluator.evaluate()
            result.update(metrics)
            mAP.append(metrics['PascalBoxes_Precision/mAP@{}IOU'.format(iou)])
        return mAP, result