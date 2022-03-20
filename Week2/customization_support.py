import sys
import numpy as np
import cv2, os
from PIL import Image
from detectron2.structures import BoxMode


def getItemsFromMask(maskPath):
    """
    Reads the mask files in the KITTI-MOTS set and returns a list of objects in the format:
    'box': list of four corners of the box
    'class_id': 1 for car, 2 for pedestrian
    'poly': list of tuples containing point coordinates of shape
    """
    mask = np.array(Image.open(maskPath))
    obj_ids = np.unique(mask)

    objs = []
    for obj in obj_ids[1:]:
        if obj != 10000:
            maskAux = np.zeros(np.shape(mask))
            maskAux[mask == obj] = obj
            maskAux = maskAux.astype(np.uint8)

            counts, hier = cv2.findContours(maskAux, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)

            for cont in counts:
                pxs = [p[0][0] for p in cont]
                pys = [p[0][1] for p in cont]
                box = [np.min(pxs), np.min(pys), np.max(pxs), np.max(pys)]

                class_id = obj // 1000
                obj_instance_id = obj % 1000

                objs.append({'box':[box[0], box[1], box[2], box[3]], 'class_id':class_id, 'object_id': obj_instance_id, 'poly': list(zip(pxs,pys))})

    return objs


def get_KITTIMOTS_dicts(data_type):
    """
    Registers the KITTI-MOTS dataset to detectron2
    """
    if data_type == 'train':
        img_dir = '/home/mcv/datasets/KITTI-MOTS/instances/'
        sequences = ['0000', '0001', '0002', '0003', '0004', '0005', '0006', '0007', '0007', '0008', '0009', '0010', '0011']
    elif data_type == 'valid':
        img_dir = '/home/mcv/datasets/KITTI-MOTS/instances/'
        sequences = ['0012', '0013', '0014', '0015', '0016', '0017', '0018', '0019', '0020']

    listOfFiles = list()
    for sequence in sequences:
        for filename in os.listdir(img_dir + sequence):
            listOfFiles.append(img_dir + sequence + '/' + filename)

    dataset_dicts = []
    for idx, filename in enumerate(listOfFiles):
        if filename[-3:] == 'png':
            record = {}
            
            height, width = cv2.imread(filename).shape[:2]
            
            record["file_name"] = filename
            record["image_id"] = idx
            record["height"] = height
            record["width"] = width
        
            objs = []
            boxes = getItemsFromMask(filename)
            for elems in boxes:
                if elems['class_id'] != 10:
                    # poly = [p for x in elems['poly'] for p in x]
                    obj = {
                        "bbox": elems['box'],
                        "bbox_mode": BoxMode.XYXY_ABS,
                        # "segmentation": [poly],
                        "category_id": elems['class_id'] - 1, #?????
                    }
                    objs.append(obj)
            record["annotations"] = objs
            dataset_dicts.append(record)
    return dataset_dicts