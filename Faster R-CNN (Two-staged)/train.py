#!/usr/bin/env python
# coding: utf-8

# # Tutorial adapted from the Detectron2 colab example

# # Install detectron2
# https://github.com/facebookresearch/detectron2
# 
# https://detectron2.readthedocs.io/en/latest/tutorials/install.html


# check pytorch installation: 
import torch, torchvision
print(torch.__version__, torch.cuda.is_available())

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
import wandb
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.data.datasets import register_coco_instances


# # Train on a iSAID dataset

# In this section, we show how to train an existing detectron2 model on the iSAID dataset.
# 
# 
# ## Prepare the dataset

# Since iSAID is in COCO format, it can be easily registered in Detectron2

wandb.login()
wandb.init(project='detectron2',sync_tensorboard=True,settings=wandb.Settings(start_method="thread", console="off"))

register_coco_instances("iSAID_train", {}, 
                        "/apps/local/shared/CV703/datasets/iSAID/iSAID_patches/train/instancesonly_filtered_train.json",
                        "/apps/local/shared/CV703/datasets/iSAID/iSAID_patches/train/images/")
register_coco_instances("iSAID_val", {}, 
                        "/apps/local/shared/CV703/datasets/iSAID/iSAID_patches/val/instancesonly_filtered_val.json",
                        "/apps/local/shared/CV703/datasets/iSAID/iSAID_patches/val/images/")
register_coco_instances("iSAID_test", {}, 
                        "/apps/local/shared/CV703/datasets/iSAID/iSAID_patches/test/test_info.json",
                        "/apps/local/shared/CV703/datasets/iSAID/iSAID_patches/test/images/")

cfg = get_cfg()
cfg.OUTPUT_DIR = 'output_fasterrcnn'

cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("iSAID_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 6
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.MAX_ITER = 150000

cfg.SOLVER.BASE_LR = 0.0025
cfg.SOLVER.MOMENTUM = 0.9
cfg.SOLVER.WEIGHT_DECAY = 0.0001

cfg.SOLVER.GAMMA = 0.1
cfg.SOLVER.STEPS = (50000,100000)

cfg.MODEL.RPN.NMS_THRESH = 0.6
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256   # faster (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 15


os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()