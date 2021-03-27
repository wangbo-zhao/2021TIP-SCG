# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn

from .roi_box_feature_extractors import make_roi_box_feature_extractor
from .roi_box_predictors import make_roi_box_predictor
from .inference import make_roi_box_post_processor
from .loss import make_roi_box_loss_evaluator
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.structures.bounding_box import BoxList

class ROIBoxHead(torch.nn.Module):
    """
    Generic Box Head class.
    """

    def __init__(self, cfg, in_channels):
        super(ROIBoxHead, self).__init__()
        self.feature_extractor = make_roi_box_feature_extractor(cfg, in_channels)
        self.predictor = make_roi_box_predictor(
            cfg, self.feature_extractor.out_channels)
        self.post_processor = make_roi_box_post_processor(cfg)
        self.loss_evaluator = make_roi_box_loss_evaluator(cfg)



    def forward(self, features, proposals, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """

        if self.training:
            # Faster R-CNN subsamples during training the proposals with a fixed
            # positive / negative ratio
            with torch.no_grad():
                proposals = self.loss_evaluator.subsample(proposals, targets) # 512

        # extract features that will be fed to the final classifier. The
        # feature_extractor generally corresponds to the pooler + heads
        x = self.feature_extractor(features, proposals)
        # final classifier that converts the features into predictions
        class_logits, box_regression = self.predictor(x)

        if not self.training:
            _, result = self.post_processor((class_logits, box_regression), proposals)

            return x, result, {}



        loss_classifier, loss_box_reg = self.loss_evaluator(
            [class_logits], [box_regression]
        )
        #dom't do regression to gt_box in proposal

        with torch.no_grad():
            proposals_num = [proposal.bbox.shape[0] for proposal in proposals]

            positive_proposals, positive_inds = keep_only_positive_boxes(proposals)
            gt_nums = [target.bbox.shape[0] for target in targets]

            proposals_generateds = [proposal[:-gt_num] for proposal, gt_num in zip(positive_proposals, gt_nums)]
            proposals_gts = [proposal[-gt_num:] for proposal, gt_num in zip(positive_proposals, gt_nums)]

            class_logits_splits = list(class_logits.split(proposals_num, dim=0))
            box_regression_splits = list(box_regression.split(proposals_num, dim=0))

            output_proposals = []

            for proposals_generated, proposals_gt, class_logits_split, box_regression_split, gt_num, positive_ind in zip(
                    proposals_generateds, proposals_gts,
                    class_logits_splits, box_regression_splits, gt_nums, positive_inds):

                class_logits_positive = class_logits_split[positive_ind, :]
                box_regression_positive = box_regression_split[positive_ind, :]

                if proposals_generated.bbox.shape[
                    0] == 0:  # if all generated boxes are not postive(here boxlist is 0),so we will only use gt proposals
                    output_proposal = proposals_gt
                    output_proposals.append(output_proposal)

                else:
                    # a = self.box_coder.decode(box_regression_positive[:-gt_num, :], proposals_generated.bbox)
                    # a = a.reshape(-1, 4)

                    before_processproposal, _ = self.post_processor(
                        (class_logits_positive[:-gt_num, :], box_regression_positive[:-gt_num, :]),
                        [proposals_generated])
                    before_processproposal = before_processproposal[0]
                    boxes = before_processproposal.bbox[
                        [i for i in range(before_processproposal.bbox.shape[0]) if i % 2 == 1]]
                    # get the box that is similare to ground truth
                    proposal = BoxList(boxes, proposals_gt.size, mode="xyxy")
                    proposal.add_field("scores", torch.ones(boxes.shape[0]).float().cuda())
                    proposal.add_field("labels", torch.ones(boxes.shape[0]).long().cuda())

                    proposals_gt = BoxList(proposals_gt.bbox, proposals_gt.size, mode="xyxy")
                    proposals_gt.add_field("scores", torch.ones(gt_num).float().cuda())
                    proposals_gt.add_field("labels", torch.ones(gt_num).long().cuda())
                    output_proposal = cat_boxlist([proposal, proposals_gt])
                    output_proposals.append(output_proposal)

        return (
            x,
            output_proposals,
            dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg),
        )

def build_roi_box_head(cfg, in_channels):
    """
    Constructs a new box head.
    By default, uses ROIBoxHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIBoxHead(cfg, in_channels)

def keep_only_positive_boxes(boxes):
    """
    Given a set of BoxList containing the `labels` field,
    return a set of BoxList for which `labels > 0`.

    Arguments:
        boxes (list of BoxList)
    """
    assert isinstance(boxes, (list, tuple))
    assert isinstance(boxes[0], BoxList)
    assert boxes[0].has_field("labels")
    positive_boxes = []
    positive_inds = []
    num_boxes = 0
    for boxes_per_image in boxes:
        labels = boxes_per_image.get_field("labels")
        inds_mask = labels > 0
        inds = inds_mask.nonzero().squeeze(1)
        positive_boxes.append(boxes_per_image[inds])
        positive_inds.append(inds_mask)
    return positive_boxes, positive_inds