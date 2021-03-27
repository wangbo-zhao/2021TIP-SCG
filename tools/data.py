import pickle as pkl
import cv2
import numpy as np

all_dataset_dir = "/home/zhaowangbo/ablation study/maskrcnn-benchmark/tools/dataset.pkl"

with open(all_dataset_dir, "rb") as f:
    dataset = pkl.load(f)



for i in range(300):
    print(i)
    image = dataset['test_imgs'][i % 300]
    # gt_salient = dataset["train_gt_salients"][i % 500][0] *255
    # gt_edges = (1 - dataset["train_edges"][i % 500]) * 255  # [h, w]

    cv2.imwrite("image/{}.jpg".format(i), image)
    # cv2.imwrite("salient/{}.png".format(i), gt_salient)
    # cv2.imwrite("edge/{}.jpg".format(i), gt_edges)
