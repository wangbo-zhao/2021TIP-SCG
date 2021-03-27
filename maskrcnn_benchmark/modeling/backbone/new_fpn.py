import torch
from . import fpn as fpn_module
from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.make_layers import conv_with_kaiming_uniform
from collections import OrderedDict
from torch import nn




def build_resnet_fpn_backbone(cfg):

    in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS  # 256
    out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS  # 256
    fpn = fpn_module.FPN(
        in_channels_list=[
            in_channels_stage2,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU
        ),
        top_blocks=fpn_module.LastLevelMaxPool(),
    )
    model = nn.Sequential(OrderedDict([("fpn", fpn)]))
    model.out_channels = out_channels
    return model





def build_new_fpn(cfg):
    assert cfg.MODEL.BACKBONE.CONV_BODY in registry.BACKBONES, \
        "cfg.MODEL.BACKBONE.CONV_BODY: {} are not registered in registry".format(
            cfg.MODEL.BACKBONE.CONV_BODY
        )
    return build_resnet_fpn_backbone(cfg)
