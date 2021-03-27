# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os

import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir

# Check if we can enable mixed-precision via apex.amp
try:
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for mixed precision via apex.amp')


def main(model_dir):
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument(
        "--config-file",
        default="/private/home/fmassa/github/detectron.pytorch_v2/configs/e2e_faster_rcnn_R_50_C4_1x_caffe2.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--ckpt",
        help="The path to the checkpoint for test, default is the latest checkpoint.",
        default=None,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    save_dir = ""
    logger = setup_logger("maskrcnn_benchmark", save_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(cfg)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    model = build_detection_model(cfg)
    model.to(cfg.MODEL.DEVICE)

    # Initialize mixed-precision if necessary
    use_mixed_precision = cfg.DTYPE == 'float16'
    amp_handle = amp.init(enabled=use_mixed_precision, verbose=cfg.AMP_VERBOSE)

    output_dir = cfg.OUTPUT_DIR
    checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir) # TODO change the pth
    ckpt = cfg.MODEL.WEIGHT if args.ckpt is None else args.ckpt

    # model_dir = "/home/zhaowangbo/modify3/salient_segmentation_fuse/salient_segmentaion_fuse/tools/model_0062500.pth"

    _ = checkpointer.load(model_dir)

    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder

    map050, map055, map060, map065, map070, map075, map080, map085, map090, map095, map = inference(model)
    return map050, map055, map060, map065, map070, map075, map080, map085, map090, map095, map


if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "8"
    results = {}
    # model = ["50000.pth", "52500.pth", "55000.pth", "57500.pth", "60000.pth", "62500.pth", "65000.pth", "67500.pth", "70000.pth"]
    # model = ["40000.pth", "42500.pth", "45000.pth", "47500.pth"]
    #          "72500.pth", "75000.pth"]
    model = ["55000.pth"]
    # model = ["55000.pth"]
    # model = ["55000.pth"]
    # model = ["65000.pth", "67500.pth", "70000.pth"]
    # model = ["45000.pth", "47500.pth"]
    # model = ["00000.pth", "07500.pth", "10000.pth", "12500.pth", "15000.pth", "20000.pth", "20000.pth"]
    # model = ["ceshi.pth"]

    for i in model:
        print("start %s" % i)
        result = {"map0.50": None, "map0.55": None, "map0.60": None, "map0.65": None, "map0.70": None, "map0.75": None,
                  "map0.80": None, "map0.85": None, "map0.90": None, "map0.95": None, "map":None}

        model_dir = "/home/zhaowangbo/SCG/SCG_TIP/base14/tools/model_00" + i
        map050, map055, map060, map065, map070, map075, map080, map085, map090, map095, map = main(model_dir)
        result["map0.50"] = map050
        result["map0.55"] = map055
        result["map0.60"] = map060
        result["map0.65"] = map065
        result["map0.70"] = map070
        result["map0.75"] = map075
        result["map0.80"] = map080
        result["map0.85"] = map085
        result["map0.90"] = map090
        result["map0.95"] = map095
        result["map"] = map

        print(result)

        results.update({i: result})

    print(results)