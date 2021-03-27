# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import time
import os

import torch
from tqdm import tqdm

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data.datasets.evaluation import evaluate
from ..utils.comm import is_main_process, get_world_size
from ..utils.comm import all_gather
from ..utils.comm import synchronize
from ..utils.timer import Timer, get_time_str
from .bbox_aug import im_detect_bbox_aug
import os.path as osp
import pickle as pkl
import numpy as np
import cv2
from evaluation import eval
import copy
import time
 #TODO when test:change the dataset DIR and change the numbmer or interatciton. And change show to FALSE
 #TODO AND change the nms in box_head to 0.7 when test and to 0.4 when show
image_h = 480
image_w = 640
show = True




dataset_ILSO_dir = "/home/zhaowangbo/SCG/ablation study/maskrcnn-benchmark/tools/dataset.pkl"
dataset_SOC_dir = "/data/zhaowangbo/salient_instance/dataset_soc_test.pkl"

color_table = np.array([[0, 215, 255], [0, 69, 255], [47, 255, 173], [204, 209, 72], [250, 206, 135], [193, 182, 255], [219, 112, 147]]*10)


class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.
        self.warm_up = 0

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        if self.warm_up < 100:
            self.warm_up += 1
            return self.diff
        else:
            self.total_time += self.diff
            self.calls += 1
            self.average_time = self.total_time / self.calls

        if average:
            return self.average_time
        else:
            return self.diff

class Dataset:
    def __init__(self, isTraining=True):
        with open(dataset_ILSO_dir, 'rb') as f:
            dataset = pkl.load(f) # ['train_segs', 'train_imgs', 'test_imgs', 'test_segs', 'test_boxes', 'train_boxes'])

        if isTraining:
            self.imgs = dataset['train_imgs']
            self.boxes = dataset['train_boxes']
            self.segs = dataset['train_segs']
        else:
            self.imgs = dataset['test_imgs']
            self.boxes = dataset['test_boxes']
            self.segs = dataset['test_segs']
        self.pos = 0
        self.trainset_size = 500
        self.isTraining = isTraining

    def forward(self):
        if self.isTraining:
            img = self.imgs[self.pos % self.trainset_size]
            gt_boxes = self.boxes[self.pos % self.trainset_size]
            gt_masks = self.segs[self.pos % self.trainset_size]
        else:
            img = self.imgs[self.pos % len(self.imgs)]
            gt_boxes = self.boxes[self.pos % len(self.boxes)]
            gt_masks = self.segs[self.pos % len(self.segs)]
        self.pos += 1

        return img, gt_boxes, gt_masks


def clip_boxes(boxes, im_shape):
    boxes[:, 0::4] = np.maximum(boxes[:, 0::4], 0)
    boxes[:, 1::4] = np.maximum(boxes[:, 1::4], 0)
    boxes[:, 2::4] = np.minimum(boxes[:, 2::4], im_shape[1] - 1)
    boxes[:, 3::4] = np.minimum(boxes[:, 3::4], im_shape[0] - 1)

    keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])

    return boxes, keep


def inference(model):
    """

     TODO modifier _C.MODEL.ROI_HEADS.DETECTIONS_PER_IMG = 20 and  box_head/inference self.detections_per_img = 20


    """
    dataset = Dataset(isTraining=False)
    model.eval()
    map_dicts = {}

    res = []
    total_time = 0
    _t = {'inference': Timer(), 'im_detect': Timer(), 'misc': Timer()}
    for i in range(300):
        print("image: ", i)

        im, gt_boxes, gt_masks = dataset.forward()

        gt_boxes[:, 2] = gt_boxes[:, 0] + gt_boxes[:, 2]
        gt_boxes[:, 3] = gt_boxes[:, 1] + gt_boxes[:, 3]


        image = im.astype(np.float32, copy=True)


        PIXEL_MEANS = (102.9801, 115.9465, 122.7717)
        PIXEL_STDS = (1, 1, 1)

        image = (image - np.array(PIXEL_MEANS)) / np.array(PIXEL_STDS)
        original_w, original_h = image.shape[1], image.shape[0]
        image = cv2.resize(image, (image_w, image_h))
        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image).float().cuda()
        start_time = time.time()

        _t['im_detect'].tic()
        predictions, salient_seg, contour_seg = model(image)
        _t['im_detect'].toc()

        end_time = time.time()
        total_time = total_time + end_time - start_time



        salient_seg = salient_seg.cpu().data.numpy()
        contour_seg = contour_seg.cpu().data.numpy()
        salient_seg = salient_seg[0, 0, :, :] * 255
        contour_seg = contour_seg[0, 0, :, :] * 255
    #
        salient_seg = cv2.resize(salient_seg, (original_w, original_h))
        contour_seg = cv2.resize(contour_seg, (original_w, original_h))
    #
        cv2.imwrite("salient/{}.png".format(i), salient_seg)
        cv2.imwrite("contour/{}.png".format(i), contour_seg)

        predictions = predictions[0]

        bbox = predictions.bbox
        bbox = bbox.cpu().data.numpy()

        seg_pred = predictions.extra_fields["mask"]
        seg_pred = seg_pred.cpu().data.numpy()

        cls_scores = predictions.extra_fields["scores"]
        cls_scores = cls_scores.cpu().data.numpy()

        if show:
            keep = cls_scores > 0.6
            cls_scores = cls_scores[keep]
            bbox = bbox[keep, :]
            seg_pred = seg_pred[keep, :, :]


        bbox[:, 0] *= (original_w / image_w)
        bbox[:, 1] *= (original_h / image_h)
        bbox[:, 2] *= (original_w / image_w)
        bbox[:, 3] *= (original_h / image_h)

        bbox, keep = clip_boxes(bbox, im.shape)
        # print(keep)
        # only keep the box w>0 and h>0
        bbox = bbox[keep, :]
        seg_pred = seg_pred[keep, :]
        # print(cls_scores)
        cls_scores = cls_scores[keep]



        cls_dets = np.hstack((bbox, cls_scores[:, np.newaxis])) \
            .astype(np.float32, copy=False)

        segmaps = np.zeros([len(seg_pred), im.shape[0], im.shape[1]])
        img_for_show = copy.deepcopy(im)

        for j in range(seg_pred.shape[0]):
            x0 = int(bbox[j, 0])
            y0 = int(bbox[j, 1])
            x1 = int(bbox[j, 2]) + 1
            y1 = int(bbox[j, 3]) + 1

            segmap = seg_pred[j, :, :]
            segmap = cv2.resize(segmap, (x1-x0, y1-y0),
                                interpolation=cv2.INTER_LANCZOS4)

            segmaps[j, y0:y1, x0:x1] = segmap

            if show:
                color = color_table[j]
                img_for_show[segmaps[j, :, :] > 0.5] = 0.6 * color + 0.4 * img_for_show[segmaps[j, :, :] > 0.5]

        res.append({"gt_boxes": gt_boxes, 'gt_masks': gt_masks, 'segmaps': segmaps, "bboxes": bbox, 'scores': cls_scores,
             'img': im})

        if show:
            if len(cls_dets) > 0:

                for record in cls_dets:
                    x0 = record[0]
                    y0 = record[1]
                    x1 = record[2]
                    y1 = record[3]
                    score = record[4]

                    cv2.rectangle(img_for_show, (int(x0), int(y0)), (int(x1), int(y1)), (73, 196, 141), 2)
                    # if (y0 > 10):
                    #     cv2.putText(img_for_show, str(score), (int(x0), int(y1 - 6)), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    #                 0.8, (0, 255, 0))
                    # else:
                    #     cv2.putText(img_for_show, str(score), (int(x0), int(y1 + 15)), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    #                 0.8, (0, 255, 0))
        gt = np.zeros([gt_masks[0].shape[0], gt_masks[0].shape[1], 3], gt_masks[0].dtype)
        for k in range(len(gt_masks)):
            mask = gt_masks[k].reshape([gt_masks[k].shape[0], gt_masks[k].shape[1], 1])
            gt += mask * color_table[k].reshape([1, 1, 3])

        cv2.imwrite('seg/{:d}.jpg'.format(i), img_for_show)
        cv2.imwrite("gt_soc/{:d}.png".format(i), gt)
        cv2.imwrite("img/{:d}.jpg".format(i), im)


        print("start eval")
    print("total time:", _t['im_detect'].average_time)


    map050 = eval(res, 0.50)
    map055 = eval(res, 0.55)
    map060 = eval(res, 0.60)
    map065 = eval(res, 0.65)
    map070 = eval(res, 0.70)
    map075 = eval(res, 0.75)
    map080 = eval(res, 0.80)
    map085 = eval(res, 0.85)
    map090 = eval(res, 0.90)
    map095 = eval(res, 0.95)

    map = (map050 + map055 + map060 + map065 + map070 + map075 + map080 + map085 + map090 + map095) / 10
    print(map)

    return map050, map055, map060, map065, map070, map075, map080, map085, map090, map095, map




