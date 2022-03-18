from detectron2.utils.logger import setup_logger

setup_logger

from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer

import os
import pickle
import random
import cv2
import matplotlib.pyplot as plt

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.config import get_cfg
from detectron2 import model_zoo

def plot_samples(dataset_name, n=1):
        dataset_custom = DatasetCatalog.get(dataset_name)
        dataset_custom_metadata = MetadataCatalog.get(dataset_name)

        for s in random.sample(dataset_custom, n):
            img = cv2.imread(s['file_name'])
            v = Visualizer(img[:,;,::-1], metadata=dataset_custom_metadata, scale=0.5)
            v = v.draw_dataset_dict(s)
            plt.figure(figsize=(15,20))
            plt.imshow(v.get_image())
            plt.show

from detectron2.data.datasets import register_coco_instances
register_coco_instances("my_dataset", {}, "json_annotation.json", "path/to/image/dir")

from detectron2.data import DatasetCatalog
DatasetCatalog.register("my_dataset", my_dataset_function)