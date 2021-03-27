# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time

import torch
import torch.distributed as dist

from maskrcnn_benchmark.utils.comm import get_world_size
from maskrcnn_benchmark.utils.metric_logger import MetricLogger
from apex import amp

import numpy as np
import os.path as osp
import pickle as pkl
import os
import torch
from PIL import Image
import torchvision
import cv2
from maskrcnn_benchmark.structures.bounding_box import BoxList
import random

datasetdir="/data/zhaowangbo/salient_instance/dataset_soc+ilso_train_withsalient.pkl"
dataset_ILSO_dir = "/data/zhaowangbo/salient_instance/dataset_ilso_train_withsalient.pkl"
DUTS_datasetdir = "/data/zhaowangbo/salient_instance/DUTS/DUTS-TR/dataset_duts_withedge.pkl"
all_dataset = "/data/zhaowangbo/salient_instance/dataset_ilso_withall.pkl"
BSDS500_datasetdir = "/data/zhaowangbo/salient_instance/dataset_bsds500_train.pkl"
PASCAL_datasetdir = "/data/zhaowangbo/salient_instance/dataset_pascal_train.pkl"

"""dataset_DUTS_train.pkl
train_imgs [h, w, 3] BGR
train_salients [h, w] 0,1
train_edge [h, w] 0, 1

"""
"""
dataset_ilso_withall.pkl
train_imgs [h, w, 3] BGR
train_edges [h, w] 0,1
"""
"""
dataset_bsds500_train.pkl
trian_imgs[h, w, 3] BGR
train_edges[h, w] 0,1
"""
"""
dataset_pascal_train.pkl
num=
trian_imgs[h, w, 3] BGR
train_edges[h, w] 0,1
"""


trainset_size = 500
DUTS_trainset_size = 10553
BSDS500_trainset_size = 500
PASCAL_trainset_size = 20206
num_interation = 70000
image_h = 480
image_w = 640

def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def random_flip(image, gt_boxes, gt_masks, gt_salient, gt_edges):

    w = image.shape[2]
    if random.randint(0, 1000) % 2 == 0:
        return image, gt_boxes, gt_masks, gt_salient, gt_edges


    image = image[:, :, ::-1]

    oldx1 = gt_boxes[:, 0].copy()
    oldx2 = gt_boxes[:, 2].copy()
    gt_boxes[:, 0] = w - oldx2
    gt_boxes[:, 2] = w - oldx1
    gt_masks = gt_masks[:, :, ::-1] #[n, h, w]

    gt_salient = gt_salient[:, :, ::-1] #[1, h, w]
    gt_edges = gt_edges[:, ::-1] #[h, w]

    return image.copy(), gt_boxes.copy(), gt_masks.copy(), gt_salient, gt_edges



#image[3, h, w], gt_boxes[n, 4], gt_mask[n, h, w], gt_salient[1, h, w]
def random_crop(image, gt_boxes, gt_mask, gt_salient, gt_edges):


    _, h_img, w_img = image.shape

    max_bbox = np.concatenate([np.min(gt_boxes[:, 0:2], axis=0), np.max(gt_boxes[:, 2:4], axis=0)], axis=-1)
    max_l_trans = max_bbox[0]
    max_u_trans = max_bbox[1]
    max_r_trans = w_img - max_bbox[2]
    max_d_trans = h_img - max_bbox[3]

    crop_xmin = max(0, int(max_bbox[0] - random.randint(0, max_l_trans)))
    # crop_xmin = min(crop_xmin, 16)

    crop_ymin = max(0, int(max_bbox[1] - random.randint(0, max_u_trans)))
    # crop_ymin = min(crop_ymin, 16)

    crop_xmax = max(w_img, int(max_bbox[2] + random.randint(0, max_r_trans)))
    # crop_xmax = max(crop_xmax, w_img - 16)

    crop_ymax = max(h_img, int(max_bbox[3] + random.randint(0, max_d_trans)))
    # crop_ymax = max(crop_ymax, h_img - 16)

    image = image[:, crop_ymin: crop_ymax, crop_xmin: crop_xmax]
    gt_mask = gt_mask[:, crop_ymin:crop_ymax, crop_xmin:crop_xmax]
    gt_salient = gt_salient[:, crop_ymin:crop_ymax, crop_xmin:crop_xmax] #[1, h, w]
    gt_edges = gt_edges[crop_ymin:crop_ymax, crop_xmin:crop_xmax] # [h,w]

    gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] - crop_xmin
    gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] - crop_ymin

    w_origin = image.shape[2]
    h_origin = image.shape[1]

    img = np.zeros([image.shape[0], image_h, image_w], dtype=np.float32)
    for i in range(image.shape[0]):
        img[i] = cv2.resize(image[i], (image_w, image_h))

    image = img


    mask = np.zeros([gt_mask.shape[0], image_h, image_w], dtype=np.uint8)
    for i in range(gt_mask.shape[0]):
        mask[i] = cv2.resize(gt_mask[i], (image_w, image_h))

    gt_mask = mask

    gt_salient = cv2.resize(gt_salient[0,:, :], (image_w, image_h))
    gt_edges = cv2.resize(gt_edges, (image_w, image_h))




    gt_boxes[:, 0] = gt_boxes[:, 0] * (image_w / w_origin)
    gt_boxes[:, 1] = gt_boxes[:, 1] * (image_h / h_origin)
    gt_boxes[:, 2] = gt_boxes[:, 2] * (image_w  / w_origin)
    gt_boxes[:, 3] = gt_boxes[:, 3] * (image_h  / h_origin)


    return image, gt_boxes, gt_mask, gt_salient, gt_edges


def make_salient_data(img_origin, gt_salient, gt_edge):

    img_origin = img_origin.astype(np.float32)
    # img_origin = img_origin[:, :, ::-1]

    PIXEL_MEANS = (102.9801, 115.9465, 122.7717)
    PIXEL_STDS = (1, 1, 1)

    img_origin = img_origin - np.array(PIXEL_MEANS)
    img_origin = img_origin / np.array(PIXEL_STDS)





    if random.randint(0, 1000) % 2 == 0:
        img_origin = cv2.resize(img_origin, (image_w, image_h))
        gt_salient = cv2.resize(gt_salient, (image_w, image_h))
        gt_edge = cv2.resize(gt_edge, (image_w, image_h))
        # gt_edge = cv2.resize(gt_edge, (image_w, image_h))
        img_origin = img_origin.transpose(2, 0, 1) #to [3, h, w]
        return img_origin, gt_salient, gt_edge
  #img_origin [h, w, 3]
    img_origin = img_origin[:, ::-1, :] #random flip
    gt_salient = gt_salient[:, ::-1]
    gt_edge = gt_edge[:, ::-1]
    img_origin = cv2.resize(img_origin, (image_w, image_h))
    gt_salient = cv2.resize(gt_salient, (image_w, image_h))
    gt_edge = cv2.resize(gt_edge, (image_w, image_h))
    img_origin = img_origin.transpose(2, 0, 1) #to [3, h, w]
    return img_origin, gt_salient, gt_edge


def make_edge_data(img_origin, gt_edge):

    img_origin = img_origin.astype(np.float32)
    # img_origin = img_origin[:, :, ::-1]

    PIXEL_MEANS = (102.9801, 115.9465, 122.7717)
    PIXEL_STDS = (1, 1, 1)

    img_origin = img_origin - np.array(PIXEL_MEANS)
    img_origin = img_origin / np.array(PIXEL_STDS)





    if random.randint(0, 1000) % 2 == 0:
        img_origin = cv2.resize(img_origin, (image_w, image_h))
        gt_edge = cv2.resize(gt_edge, (image_w, image_h))
        # gt_edge = cv2.resize(gt_edge, (image_w, image_h))
        img_origin = img_origin.transpose(2, 0, 1) #to [3, h, w]
        return img_origin, gt_edge
  #img_origin [h, w, 3]
    img_origin = img_origin[:, ::-1, :] #random flip
    gt_edge = gt_edge[:, ::-1]
    # gt_edge = gt_edge[:, ::-1]
    img_origin = cv2.resize(img_origin, (image_w, image_h))
    gt_edge = cv2.resize(gt_edge, (image_w, image_h))
    # gt_edge = cv2.resize(gt_edge, (image_w, image_h))
    img_origin = img_origin.transpose(2, 0, 1) #to [3, h, w]
    return img_origin, gt_edge


def make_data(img_origin, boxes, gt_masks, gt_salient, gt_edges):
    img_origin = img_origin.astype(np.float32)
    # img_origin = img_origin[:, :, ::-1] #RGB to BGR


    PIXEL_MEANS = (102.9801, 115.9465, 122.7717)
    PIXEL_STDS = (1, 1, 1)

    img_origin = img_origin - np.array(PIXEL_MEANS)
    img_origin = img_origin / np.array(PIXEL_STDS)

    # w_origin, h_origin = img_origin.shape[1], img_origin.shape[0]

    # img_input = cv2.resize(img_origin, (image_w, image_h)) # resieze image to640*480 #cv2.resize(image, (w, h))
    img_input = img_origin.transpose(2, 0, 1) #transpose img from [h, w , 3] to [3, h, w]

    # gt_salient = cv2.resize(gt_salient[0,:, :], (image_w, image_h))

    # gt_boxes orgin[x0, y0, w, h] -> gt_boxes now [x0, y0, x1, y1]
    boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
    boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
    # boxes[:, 0] = boxes[:, 0] * (image_w / w_origin)
    # boxes[:, 1] = boxes[:, 1] * (image_h / h_origin)
    # boxes[:, 2] = boxes[:, 2] * (image_w / w_origin)
    # boxes[:, 3] = boxes[:, 3] * (image_h / h_origin)

    # gt_masks = np.zeros([segm.shape[0], image_h, image_w], dtype=np.uint8) # [n, 480, 640]
    # for i in range(segm.shape[0]):
    #     gt_masks[i] = cv2.resize(segm[i], (image_w, image_h))

    return img_input, boxes, gt_masks, gt_salient, gt_edges



def do_train(model, optimizer, scheduler, checkpointer, checkpoint_period, arguments):


    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")


    model.train()

    with open(all_dataset, 'rb') as f:
        dataset = pkl.load(f)
    print("loading instance dataset")

    with open(DUTS_datasetdir, "rb") as f:
        salient_dataset = pkl.load(f)
    print("loading salient dataset")

    with open(PASCAL_datasetdir, "rb") as f:
        edge_dataset = pkl.load(f)
    print("loading edge dataset")
    print(len(edge_dataset["train_imgs"]))

    start_training_time = time.time()
    end = time.time()
    for i in range(0, num_interation+1):
########################################################################################################################
        # train instance stage
        model.train_stage = "instance"

        image = dataset['train_imgs'][i % trainset_size]
        gt_boxes = dataset["train_boxes"][i % trainset_size].copy()
        gt_masks = dataset["train_segs"][i % trainset_size]
        gt_salient = dataset["train_gt_salients"][i % trainset_size]
        gt_edges = dataset["train_edges"][i % trainset_size] #[h, w]


        iteration = i + 1
        arguments["iteration"] = iteration
        scheduler.step()


        image, boxes, gt_masks, gt_salient, gt_edges = make_data(image, gt_boxes, gt_masks, gt_salient, gt_edges)
        image, boxes, gt_masks, gt_salient, gt_edges = random_flip(image, boxes, gt_masks, gt_salient, gt_edges)
        image, boxes, gt_masks, gt_salient, gt_edges = random_crop(image, boxes, gt_masks, gt_salient, gt_edges)

        gt_salient = gt_salient[np.newaxis, :, :]
        gt_salient = np.ascontiguousarray(gt_salient, dtype=np.long)
        gt_edges = gt_edges[np.newaxis, :, :]
        gt_edges = np.ascontiguousarray(gt_edges, dtype=np.long)



        classes = [1] * len(boxes)
        classes = torch.tensor(classes).cuda()

        boxes = torch.as_tensor(boxes).reshape(-1, 4).cuda()
        target = BoxList(boxes, (image_w, image_h), mode="xyxy") # # (image_width, image_height)

        image = torch.from_numpy(image).cuda()
        gt_masks = torch.from_numpy(gt_masks)

        gt_salient = torch.from_numpy(gt_salient).cuda()
        gt_edges = torch.from_numpy(gt_edges).cuda()

        target.add_field("labels", classes)
        target.add_field("mask", gt_masks)
        target.add_field("gt_salient", gt_salient)
        target.add_field("gt_edge", gt_edges)

        target = [target]
        data_time = time.time() - end

        loss_dict = model(image, target)

        losses = sum(loss for loss in loss_dict.values())
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        optimizer.zero_grad()

        with amp.scale_loss(losses, optimizer) as scaled_losses:
            scaled_losses.backward()
        optimizer.step()
        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)
        eta_seconds = meters.time.global_avg * (50000 - i)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if i % 20 == 0 or i == 50000:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=i,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
        if i % checkpoint_period == 0:
            checkpointer.save("model_{:07d}".format(i), **arguments)
        if i == 80000:
            checkpointer.save("model_final", **arguments)

#######################################################################################################################
        # training salient stage
        model.train_stage = "salient"
        image = salient_dataset['train_imgs'][i % DUTS_trainset_size]
        gt_salient =  salient_dataset["train_salients"][i % DUTS_trainset_size]
        gt_edge = salient_dataset["train_edges"][i % DUTS_trainset_size]

        image, gt_salient, gt_edge = make_salient_data(image, gt_salient, gt_edge)
        gt_salient = gt_salient[np.newaxis, :, :]
        gt_salient = np.ascontiguousarray(gt_salient, dtype=np.long)

        gt_edge = gt_edge[np.newaxis, :, :]
        gt_edge = np.ascontiguousarray(gt_edge, dtype=np.long)

        image = torch.from_numpy(image).cuda()
        gt_salient = torch.from_numpy(gt_salient).cuda()
        gt_edge = torch.from_numpy(gt_edge).cuda()


        boxes = torch.ones(1, 4)
        boxes = boxes.cuda()
        target = BoxList(boxes, (image_w, image_h), mode="xyxy")
        target.add_field("gt_salient", gt_salient)
        target.add_field("gt_edge", gt_edge)
        target = [target]

        loss_dict = model(image, target)

        losses = sum(loss for loss in loss_dict.values())


        optimizer.zero_grad()

        with amp.scale_loss(losses, optimizer) as scaled_losses:
            scaled_losses.backward()
        optimizer.step()
        if i % 20 == 0:
            print("salient stage: {}".format(loss_dict["loss_salient_segmentation"]))
            # print("edge_stage: {}".format(loss_dict["loss_edge"]))

########################################################################################################################
        # training edge stage
        model.train_stage = "edge"
        image = edge_dataset['train_imgs'][i % PASCAL_trainset_size]
        gt_edge =  edge_dataset["train_edges"][i % PASCAL_trainset_size ]

        image, gt_edge = make_edge_data(image, gt_edge)

        gt_edge = gt_edge[np.newaxis, :, :]
        gt_edge = np.ascontiguousarray(gt_edge, dtype=np.long)

        image = torch.from_numpy(image).cuda()
        gt_edge = torch.from_numpy(gt_edge).cuda()


        boxes = torch.ones(1, 4)
        boxes = boxes.cuda()
        target = BoxList(boxes, (image_w, image_h), mode="xyxy")
        target.add_field("gt_edge", gt_edge)


        target = [target]

        loss_dict = model(image, target)
        losses = sum(loss for loss in loss_dict.values())


        optimizer.zero_grad()

        with amp.scale_loss(losses, optimizer) as scaled_losses:
            scaled_losses.backward()

        optimizer.step()

        if i % 20 == 0:
            print("edge stage: {}".format(loss_dict["loss_edge"]))


    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / 80000
        )
    )

