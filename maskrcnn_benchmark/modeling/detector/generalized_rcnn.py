# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn

from maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads
from ..backbone.new_fpn import build_new_fpn
from ..backbone.salient_segmentation import SalientSegmentationFPN
from ..backbone.EdgeFPN import EdgeFPN
from ..backbone.feature_fuse import FeatureFuse
from ..backbone.feature_fuse_predict import PredictFuse
from torch.nn import functional as F

class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()

        self.backbone = build_backbone(cfg)
        self.fpn = build_new_fpn(cfg)
        self.fpn_for_salient = SalientSegmentationFPN(cfg)
        self.fpn_for_edge = EdgeFPN(cfg)

        self.fuse = FeatureFuse()

        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)
        self.train_stage = "instance"



    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.train_stage == "instance":

            if self.training and targets is None:
                raise ValueError("In training mode, targets should be passed")
            images = to_image_list(images)

            features = self.backbone(images.tensors)
            origine_features = self.fpn(features)



            salient_features, salient_loss, predict_salient = self.fpn_for_salient(origine_features, targets)
            edge_features, edge_loss, predict_contour = self.fpn_for_edge(origine_features, targets)
            # salient_seg_features = F.softmax(salient_seg_features, dim=1) #[1, 2, 120, 160]
            # edge_seg_features = F.softmax(edge_seg_features, dim=1) #[1, 2, 120, 160]


            features = self.fuse(origine_features, salient_features, edge_features)


            proposals, proposal_losses = self.rpn(images, features, targets)



            if self.roi_heads:
                x, result, detector_losses = self.roi_heads(features, proposals, targets)

            else:
                # RPN-only models don't have roi_heads
                x = features
                result = proposals
                detector_losses = {}

            if self.training:
                losses = {}

                losses.update(salient_loss)
                losses.update(detector_losses)
                losses.update(proposal_losses)

                return losses


            return result, predict_salient, predict_contour

        elif self.train_stage == "salient":


            if self.training and targets is None:
                raise ValueError("In training mode, targets should be passed")
            images = to_image_list(images)

            features = self.backbone(images.tensors)
            origine_features = self.fpn(features)

            salient_features, salient_loss, predict_salient = self.fpn_for_salient(origine_features, targets)


            if self.training:
                losses = {}
                losses.update(salient_loss)

                return losses

        elif self.train_stage == "edge":
            images = to_image_list(images)

            features = self.backbone(images.tensors)
            origine_features = self.fpn(features)

            edge_features, edge_loss, predict_contour = self.fpn_for_edge(origine_features, targets)


            if self.training:
                losses = {}
                losses.update(edge_loss)
                # losses.update(edge_loss)
                return losses
