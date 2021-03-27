# Authors: 644142239@qq.com (Ruochen Fan)
# Following https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173

import pickle as pkl
import numpy as np
import cv2
from IPython import embed
from sklearn import metrics
from numba import jit

filter_thresh = 0.5


# @jit
def calc_iou(mask_a, mask_b):
    intersection = (mask_a + mask_b >= 2).astype(np.float32).sum()
    iou = intersection / (mask_a + mask_b >= 1).astype(np.float32).sum()
    return iou

# @jit
def calc_area(box):
    x1 = box[0]
    y1 = box[1]
    x2 = box[2]
    y2 = box[3]
    area = (x2 - x1) * (y2 - y1)
    return area


def calc_box_iou(box_a, box_b):
    area_a = calc_area(box_a)
    area_b = calc_area(box_b)
    x1_intersection = max(box_a[0], box_b[0])
    y1_intersection = max(box_a[1], box_b[1])
    x2_intersection = min(box_a[2], box_b[2])
    y2_intersection = min(box_a[3], box_b[3])
    intersection = (x2_intersection - x1_intersection) * (y2_intersection - y1_intersection)

    iou = intersection / (area_a + area_b - intersection)

    return iou
# @jit
def calc_accu_recall(gt_masks, segmaps, iou_thresh, ious):
    num_TP = 0
    for i in range(len(gt_masks)):
        max_match = 0
        max_match_id = 0
        for j in range(len(segmaps)):
            if ious[i, j] > max_match:
                max_match = ious[i, j]
                max_match_id = j
        if max_match > iou_thresh:  # find the segmaps have the largest IOU with gt, if the largest > thresh,tP+1
            num_TP += 1
            ious[:, max_match_id] = 0

    recall = num_TP / len(gt_masks)
    accu = num_TP / len(segmaps)

    return recall, accu


def eval(reses, iou_thresh):
    aps = []
    aps_overlapped = []
    aps_divided = []
    print("lengh of the result", len(reses))

    for ind, res in enumerate(reses):
        gt_masks = res['gt_masks']  # [n, w, h]
        gt_boxes = res["gt_boxes"]  # [n, 4]

        segmaps = res['segmaps']
        bboxes = res["bboxes"]

        scores = res['scores']

        order = np.argsort(scores)[::-1]
        scores = scores[order]
        segmaps = segmaps[order]
        segmaps = (segmaps > filter_thresh).astype(np.int32)  # if the pixel >0.5 ,turn it to 1,else turn it to 0.

        bboxes = bboxes[order]

        ious = np.zeros([100, 100])
        for i in range(len(gt_masks)):
            for j in range(len(segmaps)):
                # ious[i, j] = calc_box_iou(gt_boxes[i], bboxes[j])
                ious[i, j] = calc_iou(gt_masks[i], segmaps[j])


        recall_accu = {}
        for i in range(len(scores)):
            accu, recall = calc_accu_recall(gt_masks, segmaps[:i + 1], iou_thresh, ious.copy())

            if recall in recall_accu:
                if accu > recall_accu[recall]:
                    recall_accu[recall] = accu
            else:
                recall_accu[recall] = accu

        recalls = list(recall_accu.keys())
        recalls.sort()
        accus = []
        for recall in recalls:
            accus.append(recall_accu[recall])
        accus = accus[:1] + accus
        recalls = [0] + recalls
        if accus == []:
            accus.append(0)
            accus.append(0)
            recalls.append(0)

        # print(ind)
        #
        # print(recalls)
        # print(accus)

        ap = metrics.auc(recalls, accus)
        aps.append(ap)

    return sum(aps) / len(aps)