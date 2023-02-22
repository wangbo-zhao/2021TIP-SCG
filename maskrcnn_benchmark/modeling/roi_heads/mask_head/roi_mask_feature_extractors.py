# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from torch import nn
from torch.nn import functional as F

from ..box_head.roi_box_feature_extractors import ResNet50Conv5ROIFeatureExtractor
from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.poolers import Pooler
from maskrcnn_benchmark.modeling.make_layers import make_conv3x3
import torch

registry.ROI_MASK_FEATURE_EXTRACTORS.register(
    "ResNet50Conv5ROIFeatureExtractor", ResNet50Conv5ROIFeatureExtractor
)


@registry.ROI_MASK_FEATURE_EXTRACTORS.register("MaskRCNNFPNFeatureExtractor")
class MaskRCNNFPNFeatureExtractor(nn.Module):
    """
    Heads for FPN for classification
    """

    def __init__(self, cfg, in_channels):
        """
        Arguments:
            num_classes (int): number of output classes
            input_size (int): number of channels of the input once it's flattened
            representation_size (int): size of the intermediate representation
        """
        super(MaskRCNNFPNFeatureExtractor, self).__init__()

        resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_MASK_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )
        input_size = in_channels
        self.pooler = pooler

        use_gn = cfg.MODEL.ROI_MASK_HEAD.USE_GN
        layers = cfg.MODEL.ROI_MASK_HEAD.CONV_LAYERS
        dilation = cfg.MODEL.ROI_MASK_HEAD.DILATION

        next_feature = input_size
        self.blocks = []
        # for layer_idx, layer_features in enumerate(layers, 1):
        #     layer_name = "mask_fcn{}".format(layer_idx)
        #     module = make_conv3x3(
        #         next_feature, layer_features,
        #         dilation=dilation, stride=1, use_gn=use_gn
        #     )
        #     self.add_module(layer_name, module)
        #     next_feature = layer_features
        #     self.blocks.append(layer_name)
        self.out_channels = 256
        self.first_conv = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
                                        nn.ReLU(inplace=True))
        self.down_conv = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1),
                                        nn.ReLU(inplace=True))

        self.tran = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1),
                                        nn.ReLU(inplace=True))

        self.fc = nn.Linear(12544, 196)
        self.fuse = nn.Sequential(nn.Conv2d(in_channels=257, out_channels=256, kernel_size=3, stride=1, padding=1),
                                  nn.ReLU(inplace=True))
        self.tran_x = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1)

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        x = self.first_conv(x)

        x_down = self.down_conv(x)
        x_down = self.tran(x_down)
        x_flat = x_down.view(x.shape[0], -1)
        x_up_flat = self.fc(x_flat)
        x_fc_predict = F.sigmoid(x_up_flat.view(x.shape[0], 1, *x.shape[2:]))

        x = self.fuse(torch.cat((self.tran_x(x), x_fc_predict), dim=1))
        return x, x_fc_predict





def make_roi_mask_feature_extractor(cfg, in_channels):
    func = registry.ROI_MASK_FEATURE_EXTRACTORS[
        cfg.MODEL.ROI_MASK_HEAD.FEATURE_EXTRACTOR
    ]
    return func(cfg, in_channels)
