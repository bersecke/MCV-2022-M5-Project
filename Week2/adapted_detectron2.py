from importlib.resources import path
import torch

# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# Import some common libraries
import numpy as np
import os, json, cv2, random

# Import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from PIL import Image

from customization_support import get_KITTIMOTS_dicts

# ------------------------------------------------------------

# Preparing the custom dataset

path_train_imgs = '../KITTI-MOTS/training/image_02/' #'/home/mcv/datasets/KITTI-MOTS/training/image_02/'
path_train_labels = '../KITTI-MOTS/instances/' #'/home/mcv/datasets/KITTI-MOTS/instances/'


for d in ['train', 'valid']:
    DatasetCatalog.register("KITTIMOTS_" + d, lambda d=d: get_KITTIMOTS_dicts(d))
    MetadataCatalog.get("KITTIMOTS_" + d).set(thing_classes=["pedestrian", "car"])
KITTIMOTS_metadata = MetadataCatalog.get("KITTIMOTS_train")

print(DatasetCatalog.list())


# Visualization



# ------------------------------------------------------------

# Training
