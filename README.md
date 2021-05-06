# SCG
🔥🔥🔥The code for SCG: Saliency and Contour Guided Salient Instance Segmentation🔥🔥🔥


## Usage
1.Build. Follow the installation instructions of [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark).

2.Train. The whole network, the saliency branch, and the contour branch will be iteratively trained.
```
cd tools
python train_net.py
```
3.Test and Eval
```
cd tools
python test_net.py
```

## Dataset
We adopt the training set of ILSO to train the Mask R-CNN part, and you can download the dataset in pickle format from [Link](https://pan.baidu.com/s/1k75LjyXCKhAAb0NWs-AhhQ)  (57ej). [DUTS](http://saliencydetection.net/duts/) and [PASCAL VOS Context](https://cs.stanford.edu/~roozbeh/pascal-context/) are adopted to train the saliency and contour branch, respectively. You can find the pre-processed data in [PoolNet](https://github.com/backseason/PoolNet).





## Pretrained model
We provide the pretrained model for SCG* in [Google Drive](https://drive.google.com/file/d/1qynfmXlQhiol_1xh4M6a-xtW-szKlsBX/view?usp=sharing) and [Baidu Drive](https://pan.baidu.com/s/1BggBtg4GJFNioRy0n5f1vQ)(plsf).

## Acknowledgement
This repository is built upon [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) and [S4Net](https://github.com/RuochenFan/S4Net).
