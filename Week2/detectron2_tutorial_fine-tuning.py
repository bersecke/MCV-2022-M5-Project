from operator import imod
import torch
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from PIL import Image
dataset_path = '/home/mcv/datasets/KITTI-MOTS/'

############################
#   Fine-Tuning
############################

# if your dataset is in COCO format, this cell can be replaced by the following three lines:
# from detectron2.data.datasets import register_coco_instances
# register_coco_instances("my_dataset_train", {}, "json_annotation_train.json", "path/to/image/dir")
# register_coco_instances("my_dataset_val", {}, "json_annotation_val.json", "path/to/image/dir")

from detectron2.structures import BoxMode


def getItemsFromMask(maskPath):
    mask_1 = cv2.imread(maskPath,cv2.IMREAD_GRAYSCALE)
    # output = cv2.connectedComponentsWithStats(mask_1, 8, cv2.CV_32S)
    # (numLabels, labels, boxes, centroids) = output
    mask = np.array(Image.open(maskPath))
    obj_ids = np.unique(mask)
    print(obj_ids)

    objs = []
    # print(boxes)
    for id in obj_ids[1:]:
        if id != 10000:
            maskAux = np.zeros(np.shape(mask))
            maskAux[mask == id] = id
            counts, hier = cv2.findContours(mask_1, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
            for cont in counts:
                pxs = [p[0][0] for p in cont]
                pys = [p[0][1] for p in cont]
                box = [np.min(pxs), np.min(pys), np.max(pxs), np.max(pys)]
                class_id = id // 1000
                objs.append({'box':[box[0], box[1], box[2], box[3]], 'class_id':class_id, 'poly': list(zip(pxs,pys))})

    return objs


def get_KITTIMOTS_dicts(img_dir):
    listOfFiles = list()
    for (dirpath, dirnames, filenames) in os.walk(img_dir):
        listOfFiles += [os.path.join(dirpath, file) for file in filenames]

    dataset_dicts = []
    for idx, filename in enumerate(listOfFiles):
        record = {}
        
        height, width = cv2.imread(filename).shape[:2]
        
        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
      
        objs = []
        boxes = getItemsFromMask(filename)
        for elems in boxes:
            # assert not anno["region_attributes"]
            # anno = anno["shape_attributes"]
            # px = anno["all_points_x"]
            # py = anno["all_points_y"]
            # poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            # poly = [p for x in poly for p in x]
            if elems['class_id'] != 10:
                poly = [p for x in elems['poly'] for p in x]
                obj = {
                    "bbox": elems['box'],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": [poly],
                    "category_id": elems[4],
                }
                objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

for d in ["train", "val"]:
    DatasetCatalog.register("balloon_" + d, lambda d=d: get_balloon_dicts("balloon/" + d))
    MetadataCatalog.get("balloon_" + d).set(thing_classes=["balloon"])
balloon_metadata = MetadataCatalog.get("balloon_train")

from detectron2.engine import DefaultTrainer

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("balloon_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()
