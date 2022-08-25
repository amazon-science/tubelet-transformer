# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Transforms and data augmentation for both image + bbox.
"""
import random

import PIL
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F

from models.detr.util.box_ops import box_xyxy_to_cxcywh
from models.detr.util.misc import interpolate
import numpy as np
import numbers
import cv2
from PIL import Image


def crop(images, target, region):

    cropped_images = [F.crop(image, *region) for image in images]

    target = target.copy()
    i, j, h, w = region

    # should we do something wrt the original size?
    target["size"] = torch.tensor([h, w])

    fields = ["labels"]

    if "boxes" in target:
        boxes = target["boxes"][:,1:]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        target["boxes"][:,1:] = cropped_boxes.reshape(-1, 4)

        target['raw_boxes'] = torch.cat((target['raw_boxes'][:,0:1],target["boxes"]),1)
        target["area"] = area
        fields.append("boxes")
        fields.append('raw_boxes')

    if "masks" in target:
        # FIXME should we update the area here if there are no boxes?
        target['masks'] = target['masks'][:, i:i + h, j:j + w]
        fields.append("masks")

    # remove elements for which the boxes or masks that have zero area
    if "boxes" in target or "masks" in target:
        # favor boxes selection when defining which elements to keep
        # this is compatible with previous implementation
        if "boxes" in target:
            # cropped_boxes = target['boxes'][:,1:].reshape(-1, 2, 2)
            # keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
            areas = target['area']
            keep = torch.where(areas > 30, True, False)
            # print("eval areas")
        else:
            keep = target['masks'].flatten(1).any(1)

        for field in fields:
            target[field] = target[field][keep]

    return cropped_images, target


def hflip(images, target):
    flipped_images = [F.hflip(image) for image in images]

    w, h = images[0].size

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"][:,1:]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["boxes"][:,1:] = boxes
        target['raw_boxes'] = torch.cat((target['raw_boxes'][:,0:1],target["boxes"]),1)

    if "masks" in target:
        target['masks'] = target['masks'].flip(-1)

    return flipped_images, target


def resize(images, target, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(images[0].size, size, max_size)


    rescaled_images = [F.resize(image, size) for image in images]


    if target is None:
        return rescaled_images, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_images[0].size, images[0].size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"][:,1:]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes"][:,1:] = scaled_boxes
        target['raw_boxes'] = torch.cat((target['raw_boxes'][:,0:1],target["boxes"]),1)

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    h, w = size
    target["size"] = torch.tensor([h, w])

    if "masks" in target:
        target['masks'] = interpolate(
            target['masks'][:, None].float(), size, mode="nearest")[:, 0] > 0.5

    return rescaled_images, target


def pad(images, target, padding):
    # assumes that we only pad on the bottom right corners
    padded_images = [F.pad(image, (0, 0, padding[0], padding[1])) for image in images]
    if target is None:
        return padded_images, None
    target = target.copy()
    # should we do something wrt the original size?
    target["size"] = torch.tensor(padded_images[0][::-1])
    if "masks" in target:
        target['masks'] = torch.nn.functional.pad(target['masks'], (0, padding[0], 0, padding[1]))
    return padded_images, target


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, imgs, target):
        region = T.RandomCrop.get_params(imgs[0], self.size)
        return crop(imgs, target, region)


class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, imgs: list, target: dict):
        w = random.randint(self.min_size, min(imgs[0].width, self.max_size))
        h = random.randint(self.min_size, min(imgs[0].height, self.max_size))
        region = T.RandomCrop.get_params(imgs[0], [h, w])
        return crop(imgs, target, region)

class RandomSizeCrop_Custom(object):

    def __init__(self, size):
        self.size = size


    def __call__(self, imgs: list, target: dict):

        if imgs[0].width < imgs[0].height:
            if imgs[0].width < self.size:
                w = imgs[0].width
            else:
                w = self.size
            h = int(w*(imgs[0].height/imgs[0].width))
        else:
            if imgs[0].height < self.size:
                h = imgs[0].height
            else:
                h = self.size
            w = int(h*(imgs[0].width/imgs[0].height))

        # w = min(w, self.size * 2)
        # h = min(h, self.size * 2)

        x1 = random.randint(0, imgs[0].width - w)
        y1 = random.randint(0, imgs[0].height - h)

        return crop(imgs, target, (y1,x1,h,w))

class Resize_Custom(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, imgs, target):
        if imgs[0].width < imgs[0].height:
            w = self.size
            h = int(self.size*(imgs[0].height/imgs[0].width))
        else:
            h = self.size
            w = int(self.size*(imgs[0].width/imgs[0].height))

        # fake crop
        crop_top = int(round((imgs[0].height - h) / 2.))
        crop_left = int(round((imgs[0].width - w) / 2.))
        return crop(imgs, target, (crop_top, crop_left, h, w))

class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, imgs, target):
        image_width, image_height = imgs[0].size
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        return crop(imgs, target, (crop_top, crop_left, crop_height, crop_width))


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, imgs, target):
        if random.random() < self.p:
            return hflip(imgs, target)
        return imgs, target


class HorizontalFlip(object):
    def __call__(self, imgs, target):
        return hflip(imgs, target)


class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, imgs, target=None):
        size = random.choice(self.sizes)
        return resize(imgs, target, size, self.max_size)


class RandomPad(object):
    def __init__(self, max_pad):
        self.max_pad = max_pad

    def __call__(self, imgs, target):
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        return pad(imgs, target, (pad_x, pad_y))


class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, imgs, target):
        if random.random() < self.p:
            return self.transforms1(imgs, target)
        return self.transforms2(imgs, target)


class ToTensor(object):
    def __call__(self, imgs, target):
        return [F.to_tensor(img) for img in imgs], target


class RandomErasing(object):

    def __init__(self, *args, **kwargs):
        self.eraser = T.RandomErasing(*args, **kwargs)

    def __call__(self, imgs, target):
        return [self.eraser(img) for img in imgs], target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, images, target=None):
        N_images = [F.normalize(image, mean=self.mean, std=self.std) for image in images]
        if target is None:
            return N_images, None
        target = target.copy()
        h, w = N_images[0].shape[-2:]
        if "boxes" in target:
            boxes = target["boxes"][:,1:]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["boxes"][:, 1:] = boxes
        return N_images, target

class NormalizeRoI(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, images, target=None):
        N_images = [F.normalize(image, mean=self.mean, std=self.std) for image in images]
        if target is None:
            return N_images, None
        return N_images, target


class ColorJitter(object):
    def __init__(self, hue_shift=20.0, sat_shift=0.1, val_shift=0.1):
        # color jitter in hsv space. H: 0~360, circular S: 0.0~1.0 V: 0.0~1.0
        self.hue_bound = int(round(hue_shift / 2))
        self.sat_bound = int(round(sat_shift * 255))
        self.val_bound = int(round(val_shift * 255))

    def __call__(self, clip, target):
        # Convert: RGB->HSV
        clip_hsv = np.zeros((len(clip), clip[0].height, clip[0].width, 3)).astype(np.int32)
        for i in range(clip_hsv.shape[0]):
            clip_hsv[i] = cv2.cvtColor(np.asarray(clip[i]), cv2.COLOR_RGB2HSV)
        clip_hsv = clip_hsv.astype(np.int32)

        # Jittering.
        hue_s = random.randint(-self.hue_bound, self.hue_bound)
        clip_hsv[..., 0] = (clip_hsv[..., 0] + hue_s + 180) % 180

        sat_s = random.randint(-self.sat_bound, self.sat_bound)
        clip_hsv[..., 1] = np.clip(clip_hsv[..., 1] + sat_s, 0, 255)

        val_s = random.randint(-self.val_bound, self.val_bound)
        clip_hsv[..., 2] = np.clip(clip_hsv[..., 2] + val_s, 0, 255)

        clip_hsv = clip_hsv.astype(np.uint8)

        # Convert: HSV->RGB
        clip_ = np.zeros((len(clip), clip[0].height, clip[0].width, 3)).astype(np.uint8)
        for i in range(clip_.shape[0]):
            clip_[i] = cv2.cvtColor(clip_hsv[i], cv2.COLOR_HSV2RGB)

        return clip_, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, images, target):
        for t in self.transforms:
            images, target = t(images, target)
        return images, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string
