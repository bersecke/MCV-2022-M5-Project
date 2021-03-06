# from msilib import sequence
import sys
import numpy as np
import cv2, os
from PIL import Image
from detectron2.structures import BoxMode

from pycocotools.mask import toBbox

sys.path.insert(0, '/home/group05/MCV-2022-M5-Project')
from mots_tools.mots_common.io import load_txt

def annotation_txt_to_objs(path):
    return load_txt(path)


def getAnnoFromTxt(Path, idx):
    """
    Reads the mask files in the KITTI-MOTS set and returns a list of objects in the format:
    'box': list of four corners of the box
    'class_id': 1 for car, 2 for pedestrian
    'poly': list of tuples containing point coordinates of shape
    """
    #example
    sequence = annotation_txt_to_objs(Path)
    objs = []
    # for objs in sequence:
    if idx in sequence:
        for obj in sequence[idx]:
            obj_dict = obj.__dict__['mask']
            # print(obj_dict)
            bbox = toBbox(obj.mask)
            size_x, size_y = obj_dict['size']
            poly = {'size': [int(size_x), int(size_y)], 'counts': str(obj_dict['counts']).encode(encoding='UTF-8')}

            objs.append({'box':[bbox[0], bbox[1], bbox[2], bbox[3]], 'class_id': int(obj.__dict__['class_id']), 'poly': poly})

    return objs

#test
# getAnnoFromTxt('/home/group05/MCV-2022-M5-Project/Week2/0000.txt', 1)

def get_KITTIMOTS_dicts(data_type, pretrained=False):
    """
    Registers the KITTI-MOTS dataset to detectron2
    """
    print('get dict from txts...')
    img_dir = '/home/mcv/datasets/KITTI-MOTS/training/image_02/'
    anno_dir = '/home/mcv/datasets/KITTI-MOTS/instances_txt/'
    if data_type == 'train':
        sequences = ['0000', '0001', '0002', '0003', '0004', '0005', '0006', '0007', '0013', '0014', '0015', '0016']
    elif data_type == 'valid':
        sequences = ['0008', '0009', '0010', '0011', '0012', '0017', '0018', '0019', '0020']


    dataset_dicts = []
    for sequence in sequences:
        listOfFiles = []
        for filename in os.listdir(img_dir + sequence):
            listOfFiles.append(img_dir + sequence + '/' + filename)

        for idx, filename in enumerate(listOfFiles):
            if filename[-3:] == 'png':
                record = {}
                
                height, width = cv2.imread(filename).shape[:2]
                
                record["file_name"] = filename
                record["image_id"] = idx
                record["height"] = height
                record["width"] = width
            
                objs = []
                anno_path = anno_dir + sequence + '.txt'
                elems = getAnnoFromTxt(anno_path, idx)
                for elem in elems:
                    if pretrained:
                        if elem['class_id'] == 1: #car
                            elem['class_id'] = 3
                        elif elem['class_id'] == 2: #person
                            elem['class_id'] = 1
                        else: #ignore or  background
                            elem['class_id'] = 80
                
                    if elem['class_id'] == 2 or elem['class_id'] == 1:
                        obj = {
                            "bbox": elem['box'],
                            "bbox_mode": BoxMode.XYWH_ABS,
                            "segmentation": elem['poly'],
                            "category_id": elem['class_id'] - 1, #DOUBLECHECK
                        }
                        objs.append(obj)
                record["annotations"] = objs
                dataset_dicts.append(record)
    # print(dataset_dicts)
    return dataset_dicts