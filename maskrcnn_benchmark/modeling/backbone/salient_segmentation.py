import torch
from torch import nn
from ..backbone.new_fpn import build_new_fpn
import torch.nn.functional as F


class SalientSegmentationFPN(nn.Module):

    def __init__(self, cfg):
        super(SalientSegmentationFPN, self).__init__()

        self.fpn = build_new_fpn(cfg)


        self.Conv0 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1)
        self.Conv1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1)
        self.Conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1)
        self.Conv3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1)
        self.Conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1)

        self.final_Conv0 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.final_Conv1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.final_Conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.final_Conv3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

        self.seg_Conv = nn.Conv2d(in_channels=256, out_channels=2, kernel_size=1)



    def forward(self, origine_features, target):




        salient_seg0 = self.Conv0(origine_features[0])

        salient_seg1 = self.Conv1(origine_features[1])
        salient_seg1 = F.interpolate(salient_seg1, scale_factor=2, mode="bilinear")

        salient_seg2 = self.Conv2(origine_features[2])
        salient_seg2 = F.interpolate(salient_seg2, scale_factor=4, mode="bilinear")

        salient_seg3 = self.Conv3(origine_features[3])
        salient_seg3 = F.interpolate(salient_seg3, scale_factor=8, mode="bilinear")

        salient_seg4 = self.Conv4(origine_features[4])
        salient_seg4 = F.interpolate(salient_seg4, size=[120, 160], mode="bilinear")

        salient_feature = salient_seg0 + salient_seg1 + salient_seg2 + salient_seg3 + salient_seg4
        salient_feature = F.relu(self.final_Conv0(salient_feature))
        salient_feature = F.relu(self.final_Conv1(salient_feature))
        salient_feature = F.relu(self.final_Conv2(salient_feature))
        salient_feature = F.relu(self.final_Conv3(salient_feature))

        # salient_seg_feature = self.seg_Conv(salient_feature)
        # salient_seg_feature = F.interpolate(salient_seg_feature, scale_factor=4, mode="bilinear")



        if self.training:



            salient_seg_feature = self.seg_Conv(salient_feature)
            salient_seg_feature = F.interpolate(salient_seg_feature, scale_factor=4, mode="bilinear")




            gt_salient = target[0].get_field("gt_salient").long()

            loss_salient_segmentation0 = F.cross_entropy(salient_seg_feature, gt_salient)
            # loss_salient_segmentation1 = F.cross_entropy(salient_seg1, gt_salient)
            # loss_salient_segmentation2 = F.cross_entropy(salient_seg2, gt_salient)
            # loss_salient_segmentation3 = F.cross_entropy(salient_seg3, gt_salient)
            # loss_salient_segmentation4 = F.cross_entropy(salient_seg4, gt_salient)

            loss = loss_salient_segmentation0 * 0.5

            # loss = loss_salient_segmentation0 + loss_salient_segmentation1 + loss_salient_segmentation2 + \
            #        loss_salient_segmentation3 + loss_salient_segmentation4
            predict_salient = F.softmax(salient_seg_feature, dim=1)[:, 1, :, :].unsqueeze(0)





            return salient_feature, dict(loss_salient_segmentation=loss), predict_salient
        # salient_seg0 = self.Conv0(salient_features[0])
        # salient_seg0 = F.interpolate(salient_seg0, scale_factor=4, mode="bilinear")
        # salient_seg_feature = self.seg_Conv(salient_feature)
        # salient_seg_feature = F.interpolate(salient_seg_feature, scale_factor=16, mode="bilinear")
        # salient_seg_feature = F.softmax(salient_seg_feature, dim=1)

        salient_seg_feature = self.seg_Conv(salient_feature)
        salient_seg_feature = F.interpolate(salient_seg_feature, scale_factor=4, mode="bilinear")
        #

        predict_salient = F.softmax(salient_seg_feature, dim=1)[:, 1, :, :].unsqueeze(0)
        return salient_feature, {}, predict_salient


