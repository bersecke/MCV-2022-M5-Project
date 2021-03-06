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

# from customization_support import get_KITTIMOTS_dicts
from customization_support_rle import get_KITTIMOTS_dicts

# ------------------------------------------------------------

# Preparing the custom dataset

path_train_imgs = '/home/mcv/datasets/KITTI-MOTS/training/image_02/'
path_train_labels = '/home/mcv/datasets/KITTI-MOTS/instances/'
path_train_labels_txt = '/home/mcv/datasets/KITTI-MOTS/instances_txt/'

SAVEPATH = './KITTIMOTS_dicts_rle_poly_txt'

for d in ['train', 'valid']:
    DatasetCatalog.register("KITTIMOTS_" + d, lambda d=d: get_KITTIMOTS_dicts(d))
    MetadataCatalog.get("KITTIMOTS_" + d).set(thing_classes=["car", "pedestrian"])

KITTIMOTS_metadata = MetadataCatalog.get("KITTIMOTS_train")


# Loading or saving KITTIMOTS dicts
saving_enabled = True
saved_KITTIMOTS_dicts = SAVEPATH + '.pkl'

if os.path.exists(saved_KITTIMOTS_dicts):
    with open(saved_KITTIMOTS_dicts, 'rb') as reader:
        print('Loading dataset dicts...')
        dataset_dicts = pickle.load(reader)
else:
    print('Generating dict for training...')
    dataset_dicts = get_KITTIMOTS_dicts('train')
    if saving_enabled == True:
        with open(saved_KITTIMOTS_dicts, 'wb') as handle:
            print('Saving dataset dicts...')
            pickle.dump(dataset_dicts, handle, protocol=pickle.HIGHEST_PROTOCOL)

# ------------------------------------------------------------

# Training

from detectron2.engine import DefaultTrainer

selected_model = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
#"COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
#"COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml" 

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(selected_model))
cfg.DATASETS.TRAIN = ("KITTIMOTS_train",)
cfg.DATASETS.VAL = ('KITTIMOTS_valid',)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 0
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(selected_model)  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.0001
cfg.SOLVER.MAX_ITER = 10000
cfg.SOLVER.STEPS = [] # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512 # (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2

cfg.INPUT.MASK_FORMAT='bitmask' # For the polys in rle

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()

# ------------------------------------------------------------

# Inference and evaluation

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set a custom testing threshold
predictor = DefaultPredictor(cfg)

dataset_dicts_val = get_KITTIMOTS_dicts('valid')

# Example of inference on relevant image sample

img_filename = '/home/mcv/datasets/KITTI-MOTS/training/image_02/0019/000074.png'
for element in dataset_dicts:
    if element['file_name'] == img_filename:
        d = element
img = cv2.imread(img_filename)
im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

outputs = predictor(im_rgb)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
v = Visualizer(im_rgb[:, :, ::-1], metadata=KITTIMOTS_metadata, scale=1.2)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
image = Image.fromarray(out.get_image()[:, :, ::-1])
image.save('detectron2_trained_with_rle.png',)

# Evaluation based on COCO metrics

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

evaluator = COCOEvaluator("KITTIMOTS_valid", output_dir="./output")
val_loader = build_detection_test_loader(cfg, "KITTIMOTS_valid")
print(inference_on_dataset(predictor.model, val_loader, evaluator))