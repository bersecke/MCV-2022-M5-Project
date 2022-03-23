from glob import glob
import torch
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from PIL import Image
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description= 'Arguments to run the inference script')
    parser.add_argument('-p', '--path', default='/home/mcv/datasets/out_of_context/', type=str, help='Absolute path to image folder')
    parser.add_argument('-e', '--extension', default='.jpg', type=str, help='Absolute path to image folder')
    parser.add_argument('-o', '--out_path', default='./results', type=str, help='Relative path to output folder')
    parser.add_argument('-m', '--model', default='COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml', type=str, help='Detectron2 Model')

    return parser.parse_args()

args = parse_args()

dataset_path = args.path

config_file = args.model

os.makedirs(args.out_path, exist_ok=True)

############################
#   INFERENCE
############################

#Then, we create a detectron2 config and a detectron2 DefaultPredictor to run inference on this image.
cfg = get_cfg()

## MASK RCNN
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file(config_file))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model

# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file)
predictor = DefaultPredictor(cfg)
filenames = [img for img in glob(dataset_path + "/*"+ args.extension)]

for ind, filename in enumerate(filenames):
    im = np.array(Image.open(filename))
    outputs = predictor(im)

    # look at the outputs. See https://detectron2.readthedocs.io/tutorials/models.html#model-output-format for specification
    print(outputs["instances"].pred_classes)
    print(outputs["instances"].pred_boxes)

    # We can use `Visualizer` to draw the predictions on the image.
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    image = Image.fromarray(out.get_image()[:, :, ::-1])
    image.save(f'{args.out_path}predicted_{Path(filename).stem}')

####################################
