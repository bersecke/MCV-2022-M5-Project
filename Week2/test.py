import tensorflow as tf
import PIL.Image as Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
from skimage.measure import label, regionprops
import cv2
from torchvision.ops import masks_to_boxes
import torch
from torchvision.io import read_image
import torchvision.transforms as transforms
import torchvision.transforms.functional as F


with open('0000.txt','r') as fd:
    lines = fd.readlines()

line0 = lines[0].split(' ')



mask_1 = cv2.imread('000000.png',cv2.IMREAD_GRAYSCALE)
output = cv2.connectedComponentsWithStats(mask_1, 8, cv2.CV_32S)
(numLabels, labels, boxes, centroids) = output

img_0 = cv2.imread('000000_t.png')

dBoxes = []
# print(boxes)
for box in boxes[1:]:
    print(f'({box[0]},  {box[1]}), ({box[0]+box[2]}, {box[1]+box[3]})')
    img_0 = cv2.rectangle(img_0, (box[0],  box[1]), (box[0] + box[2], box[1] + box[3]), (255,0,0), 2)

mask = np.array(Image.open("000000.png"))
obj_ids = np.unique(mask)

obj_id = obj_ids[1]
class_id = obj_id // 1000
obj_instance_id = obj_id % 1000
print(obj_ids,class_id, obj_instance_id)

plt.imshow(img_0)
plt.show()