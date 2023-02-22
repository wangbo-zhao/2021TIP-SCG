# SCG
🔥🔥🔥The code for SCG: Saliency and Contour Guided Salient Instance Segmentation🔥🔥🔥

## TODO
- [ ] As it is suggested in [RDPNet](https://github.com/yuhuan-wu/RDPNet), we plan to use official cocoapi to evaluate the performance, for the convenience of future work.

## Usage
1.Build. Follow the installation instructions of [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark).

2.Train. The whole network, the saliency branch, and the contour branch will be iteratively trained.
```
cd tools
python train_net.py --config-file configs/e2e_mask_rcnn_R_50_FPN_1x.yaml
```
3.Test and Eval. We adopt the same evaluation code as [S4Net](https://github.com/RuochenFan/S4Net).
```
cd tools
python test_net.py
```

## Dataset
We adopt the training set of ILSO to train the Mask R-CNN part, and you can download the dataset in pickle format from [Link](https://pan.baidu.com/s/1k75LjyXCKhAAb0NWs-AhhQ)  (57ej). [DUTS](http://saliencydetection.net/duts/) and [PASCAL VOS Context](https://cs.stanford.edu/~roozbeh/pascal-context/) are adopted to train the saliency and contour branch, respectively. You can find the pre-processed data in [PoolNet](https://github.com/backseason/PoolNet).





## Pretrained model
We provide the pretrained models for SCG and SCG* in [Google Drive](https://drive.google.com/drive/folders/1xaFgVEa8eAknmfAzusYU8X9T6AvtPoGn?usp=sharing) and [Baidu Drive](https://pan.baidu.com/s/1sIiWRrRdREzpAhrnQTFjkA)(vi0x). Please put the pretrained model in `./tools/`.

## Acknowledgement
This repository is built upon [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) and [S4Net](https://github.com/RuochenFan/S4Net).

## Contact
wangbo.zhao96@gmail.com



