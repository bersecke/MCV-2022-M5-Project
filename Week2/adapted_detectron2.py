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



# Visualization
dataset_dicts = get_KITTIMOTS_dicts('train')

for d in random.sample(dataset_dicts, 1):
    split_path = d["file_name"].split('/')
    img_filename = path_train_imgs + split_path[-2] + '/' + split_path[-1]
    # img_filename_basename = os.path.basename(d["file_name"])
    # img_filename = path_train_imgs + img_filename_basename
    print(img_filename)
    img = cv2.imread(img_filename)
    visualizer = Visualizer(img[:, :, ::-1], metadata=KITTIMOTS_metadata, scale=1.2)
    out = visualizer.draw_dataset_dict(d)
    image = Image.fromarray(out.get_image()[:, :, ::-1])
    image.save('detectron2_trained.png',)

# ------------------------------------------------------------

# Training
