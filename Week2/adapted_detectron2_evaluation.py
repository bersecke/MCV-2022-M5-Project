from asyncio import selector_events
from importlib.resources import path
import torch

# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# Import some common libraries
import numpy as np
import os, json, cv2, random
import pickle

# Import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from PIL import Image

from customization_support import get_KITTIMOTS_dicts2, get_KITTIMOTS_dicts

# ------------------------------------------------------------

# Preparing the custom dataset

path_train_imgs = '/home/mcv/datasets/KITTI-MOTS/training/image_02/'
path_train_labels = '/home/mcv/datasets/KITTI-MOTS/instances/'
path_train_labels_txt = '/home/mcv/datasets/KITTI-MOTS/instances_txt/'

coco_classes = [f'obj_{i}' for i in range(80)]
coco_classes[0] = 'person'
coco_classes[2] = 'car'

for d in ['train','valid']:
    DatasetCatalog.register(f"KITTIMOTS_{d}_pretrained", lambda d=d: get_KITTIMOTS_dicts(d, True, True))
    MetadataCatalog.get(f"KITTIMOTS_{d}_pretrained").set(thing_classes=coco_classes)
KITTIMOTS_metadata = MetadataCatalog.get("KITTIMOTS_valid_pretrained")

# Evaluation

#Then, we create a detectron2 config and a detectron2 DefaultPredictor to run inference on this image.
cfg = get_cfg()
cfg.defrost()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
# cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)


from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

evaluator = COCOEvaluator("KITTIMOTS_valid_pretrained", output_dir="./output")
val_loader = build_detection_test_loader(cfg, "KITTIMOTS_valid_pretrained")
print(inference_on_dataset(predictor.model, val_loader, evaluator))