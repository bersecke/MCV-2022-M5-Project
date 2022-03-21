from math import inf
import sys
import numpy as np
import cv2, os
from PIL import Image
from detectron2.structures import BoxMode
import torch
from typing import Any, Iterator, List, Union



def _make_array(t: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    # Use float64 for higher precision, because why not?
    # Always put polygons on CPU (self.to is a no-op) since they
    # are supposed to be small tensors.
    # May need to change this assumption if GPU placement becomes useful
    if isinstance(t, torch.Tensor):
        t = t.cpu().numpy()
    return np.asarray(t).astype("float64")

def checkPolys(polys, data,class_id):
    # print('------ on checking ----------')
    # print(polys)
    # print('---------------------------------------')
    polygons_per_instance = [_make_array(p) for p in polys]
    for polygon in polygons_per_instance:
        # print('------ one poly ----------')
        # print(polygon)
        # print('---------------------------------------')
        if len(polygon) % 2 != 0 or len(polygon) < 6:
            print(data, class_id)
            print('-----------------------------',polygon)

def getSingleBox(countours):
    polys = []
    maxX, maxY = 0, 0
    minX, minY = 100000, 100000
    for cont in countours:
        pxs = [int(p[0][0]) for p in cont]
        pys = [int(p[0][1]) for p in cont]
        if minX > np.min(pxs):
            minX = np.min(pxs)
        if minY > np.min(pys):
            minY = np.min(pys)

        if maxX < np.max(pxs):
            maxX = np.max(pxs)
        if maxY < np.max(pys):
            maxY = np.max(pys)
        polyFlat = [p for x in (list(zip(pxs,pys))) for p in x]
        if len(polyFlat) > 5:
            polys.append(polyFlat)
    return [int(minX), int(minY), int(maxX), int(maxY)], polys
    

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
        # if obj != 10000:
        maskAux = np.zeros(np.shape(mask))
        maskAux[mask == obj] = obj
        maskAux = maskAux.astype(np.uint8)

        counts, hier = cv2.findContours(maskAux, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)

        if obj == 10000:
            for cont in counts:
                pxs = [int(p[0][0]) for p in cont]
                pys = [int(p[0][1]) for p in cont] 
                box = [int(np.min(pxs)), int(np.min(pys)), int(np.max(pxs)), int(np.max(pys))]
                poly = [p for x in (list(zip(pxs,pys))) for p in x]
                if len(poly) > 5:
                    objs.append({'box':box, 'class_id':3, 'object_id': 0, 'poly': [poly] })
        else:
            box,polys = getSingleBox(counts)
            class_id = obj // 1000
            obj_instance_id = obj % 1000
            objs.append({'box':box, 'class_id':int(class_id), 'object_id': int(obj_instance_id), 'poly': polys })#'poly': list(zip(pxs,pys))})



    return objs


def cover_areas_to_ignore(image, maskPath):
    """
    Sets the "Don't care" areas to 0 values on the images to perform predictions on.
    """
    mask = np.array(Image.open(maskPath))
    image[mask == 10000] = 0

    return image


def get_KITTIMOTS_dicts(data_type, ignorecase=True, pretrained=False):
    """
    Registers the KITTI-MOTS dataset to detectron2
    """
    img_dir = '/home/mcv/datasets/KITTI-MOTS/training/image_02/'
    mask_dir = '/home/mcv/datasets/KITTI-MOTS/instances/'
    if data_type == 'train':
        sequences = ['0000', '0001','0003', '0004', '0005', '0009', '0011', '0012', '0015', '0017', '0019', '0020']
    elif data_type == 'valid':
        sequences = ['0002', '0006', '0007', '0008','0010', '0013', '0014', '0016', '0018']

    listOfFiles = list()
    listOfImages = list()
    for sequence in sequences:
        for filename in os.listdir(mask_dir + sequence):
            listOfFiles.append(mask_dir + sequence + '/' + filename)
            listOfImages.append(img_dir + sequence + '/' + filename)

    dataset_dicts = []
    for idx, filename in enumerate(listOfFiles):
        if filename[-3:] == 'png':
            record = {}
            
            height, width = cv2.imread(filename).shape[:2]
            
            record["file_name"] = listOfImages[idx]
            record["image_id"] = idx
            record["height"] = height
            record["width"] = width
        
            objs = []
            boxes = getItemsFromMask(filename)
            for elems in boxes:
                # if elems['class_id'] != 10: # USELESS CONDITION
                # poly = [p for x in elems['poly'] for p in x]
                # if pretrained:
                #     if elems['class_id'] == 1: #car
                #         elems['class_id'] = 3
                #     elif elems['class_id'] == 2: #person
                #         elems['class_id'] = 1
                #     else: #ignore or  background
                #         elems['class_id'] = 80
                # if ignorecase or elems['class_id'] != 3:
                    # checkPolys(elems['poly'],record,elems['class_id'])
                obj = {
                    "bbox": elems['box'],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    # "segmentation": elems['poly'],
                    "category_id": elems['class_id'] - 1, #DOUBLECHECK
                }
                objs.append(obj)
            record["annotations"] = objs
            dataset_dicts.append(record)
    return dataset_dicts



def get_KITTIMOTS_dicts2(data_type):
    """
    Registers the KITTI-MOTS dataset to detectron2
    """
    img_dir = '/home/mcv/datasets/KITTI-MOTS/training/image_02/'
    mask_dir = '/home/mcv/datasets/KITTI-MOTS/instances/'
    if data_type == 'train':
        sequences = ['0000', '0001','0003', '0004', '0005', '0009', '0011', '0012', '0015', '0017', '0019', '0020']
    elif data_type == 'valid':
        sequences = ['0002', '0006', '0007', '0008','0010', '0013', '0014', '0016', '0018']

    listOfFiles = list()
    listOfImages = list()
    for sequence in sequences:
        for filename in os.listdir(mask_dir + sequence):
            listOfFiles.append(mask_dir + sequence + '/' + filename)
            listOfImages.append(img_dir + sequence + '/' + filename)

    dataset_dicts = []
    for idx, filename in enumerate(listOfFiles):
        if filename[-3:] == 'png':
            record = {}
            
            height, width = cv2.imread(filename).shape[:2]
            
            record["file_name"] = listOfImages[idx]
            record["image_id"] = idx
            record["height"] = height
            record["width"] = width
        
            objs = []
            boxes = getItemsFromMask(filename)
            for elems in boxes:
                if elems['class_id'] == 1 or elems['class_id'] == 2:
                    obj = {
                        "bbox": elems['box'],
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "segmentation": elems['poly'],
                        "category_id": elems['class_id'] - 1, #DOUBLECHECK
                    }
                    objs.append(obj)
            record["annotations"] = objs
            dataset_dicts.append(record)
    return dataset_dicts