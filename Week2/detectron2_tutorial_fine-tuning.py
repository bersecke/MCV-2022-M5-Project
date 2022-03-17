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
    output = cv2.connectedComponentsWithStats(mask_1, 8, cv2.CV_32S)
    (numLabels, labels, boxes, centroids) = output
    mask = np.array(Image.open(maskPath))

    objs = []
    # print(boxes)
    for box in boxes[1:]:
        # print(f'({box[0]},  {box[1]}), ({box[0]+box[2]}, {box[1]+box[3]})')
        obj_id = mask[box[1] + (box[3] // 2), box[0] + (box[2] // 2)]
        class_id = obj_id // 1000
        obj_instance_id = obj_id % 1000
        objs.append([box[0], box[1], box[0] + box[2], box[1] + box[3], class_id, obj_instance_id])

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
            if elems[4] != 10:
                obj = {
                    "bbox": [elems[0], elems[1], elems[2], elems[3]],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": [None],
                    "category_id": elems[4],
                }
                objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

# for d in ["train", "val"]:
#     DatasetCatalog.register("balloon_" + d, lambda d=d: get_balloon_dicts("balloon/" + d))
#     MetadataCatalog.get("balloon_" + d).set(thing_classes=["balloon"])


# balloon_metadata = MetadataCatalog.get("Kitti-Motts train")