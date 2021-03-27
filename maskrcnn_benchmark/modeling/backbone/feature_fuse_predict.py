import torch
from torch import nn
from ..backbone.new_fpn import build_new_fpn
import torch.nn.functional as F


class PredictFuse(nn.Module):

    def __init__(self):
        super(PredictFuse, self).__init__()
        self.transform = nn.ModuleList([nn.Sequential(nn.Conv2d(in_channels=258, out_channels=256, stride=1, kernel_size=1),
                                                      nn.ReLU(inplace=True),
                                                      nn.Conv2d(in_channels=256, out_channels=256, stride=1, kernel_size=1))
                                        for _ in range(4)])


    def forward(self, features, predict_salient, predict_contour):
        features_for_ROI = []

        for i in range(4):
            feature = torch.cat((features[i],
                                     F.interpolate(predict_salient, size=features[i].shape[2:], mode="bilinear"),
                                     F.interpolate(predict_contour, size=features[i].shape[2:], mode="bilinear")),
                                    dim=1)

            feature = self.transform[i](feature)
            features_for_ROI.append(feature)

        return features_for_ROI



