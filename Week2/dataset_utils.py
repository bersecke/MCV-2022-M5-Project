from detectron2.utils.logger import setup_logger

setup_logger

from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer

import os
import pickle
import random
import cv2
import json 
import numpy as np
import matplotlib.pyplot as plt

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.config import get_cfg
from detectron2 import model_zoo

from test import getItemsFromMask

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

def build_json(objs, save_path):          
    # Serializing json  
    with open(save_path, "w") as outfile:
        json.dump(objs, outfile, cls=MyEncoder)
    
objs = getItemsFromMask('dataset/000003_m.png')
build_json(objs, 'dataset/example_00003.json')


def plot_samples(dataset_name, n=1):
        dataset_custom = DatasetCatalog.get(dataset_name)
        dataset_custom_metadata = MetadataCatalog.get(dataset_name)

        for s in random.sample(dataset_custom, n):
            img = cv2.imread(s['file_name'])
            v = Visualizer(img[:,:,::-1], metadata=dataset_custom_metadata, scale=0.5)
            v = v.draw_dataset_dict(s)
            plt.figure(figsize=(15,20))
            plt.imshow(v.get_image())
            plt.show

# from detectron2.data.datasets import register_coco_instances
# register_coco_instances("my_dataset", {}, "json_annotation.json", "path/to/image/dir")

# from detectron2.data import DatasetCatalog
# DatasetCatalog.register("my_dataset", my_dataset_function)