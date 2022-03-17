from binascii import rledecode_hqx
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


# with open('0000.txt','r') as fd:
#     lines = fd.readlines()

# line0 = lines[0].split(' ')



mask_1 = cv2.imread('000000.png',cv2.IMREAD_GRAYSCALE)
output = cv2.connectedComponentsWithStats(mask_1, 8, cv2.CV_32S)
(numLabels, labels, boxes, centroids) = output

img_0 = cv2.imread('000000_t.png')

dBoxes = []
# print(boxes)
for box in boxes[1:]:
    print(f'({box[0]},  {box[1]}), ({box[0]+box[2]}, {box[1]+box[3]})')
    centerX = box[0] + (box[2] // 2)
    centerY = box[1] + (box[3] // 2)
    cv2.circle(img_0,(centerX, centerY), 5, (0,255,0))
    img_0 = cv2.rectangle(img_0, (box[0],  box[1]), (box[0] + box[2], box[1] + box[3]), (255,0,0), 2)

mask = np.array(Image.open("000000.png"))
print(mask.shape)
plt.imshow(mask)
plt.show()
# np.savetxt('mask.txt',mask, delimiter=' ', fmt = '%.2f')
obj_ids = np.unique(mask)
print(obj_ids)
box = boxes[1]
centerX = box[0] + (box[2] // 2)
centerY = box[1] + (box[3] // 2)
print('Pixel inside box: ', mask[centerY,centerX])

obj_id = mask[centerY,centerX]
class_id = obj_id // 1000
obj_instance_id = obj_id % 1000
print(class_id, obj_instance_id)
	
plt.imshow(img_0)
plt.show()


# internal conversion from compressed RLE format to Python RLEs object
# def rle2bbox(rle, shape):
#     '''
#     rle: run-length encoded image mask, as string
#     shape: (height, width) of image on which RLE was produced
#     Returns (x0, y0, x1, y1) tuple describing the bounding box of the rle mask
    
#     Note on image vs np.array dimensions:
    
#         np.array implies the `[y, x]` indexing order in terms of image dimensions,
#         so the variable on `shape[0]` is `y`, and the variable on the `shape[1]` is `x`,
#         hence the result would be correct (x0,y0,x1,y1) in terms of image dimensions
#         for RLE-encoded indices of np.array (which are produced by widely used kernels
#         and are used in most kaggle competitions datasets)
#     '''
    
#     a = np.fromiter(rle.split(), dtype=np.uint)
#     a = a.reshape((-1, 2))  # an array of (start, length) pairs
#     a[:,0] -= 1  # `start` is 1-indexed
    
#     y0 = a[:,0] % shape[0]
#     y1 = y0 + a[:,1]
#     if np.any(y1 > shape[0]):
#         # got `y` overrun, meaning that there are a pixels in mask on 0 and shape[0] position
#         y0 = 0
#         y1 = shape[0]
#     else:
#         y0 = np.min(y0)
#         y1 = np.max(y1)
    
#     x0 = a[:,0] // shape[0]
#     x1 = (a[:,0] + a[:,1]) // shape[0]
#     x0 = np.min(x0)
#     x1 = np.max(x1)
    
#     if x1 > shape[1]:
#         # just went out of the image dimensions
#         raise ValueError("invalid RLE or image dimensions: x1=%d > shape[1]=%d" % (
#             x1, shape[1]
#         ))

#     return x0, y0, x1, y1

# rle = '\Xe<3b;4L4M3N0O001O010O001O10OeFA_7`0`HA_7?bHB\7?cHB]7=cHD\7=bHE\7<dHFY7;gHFW7<TGAY15a7i0ZHYO\7S1_HPO[7V1cHlNY7X1dHkNZ7W1bHmN\7U1aHnN[7U1aHPO[7T1^HSO`7o0ZHWOe7j0WHYOi7[2N1002O0O2O0O2I6D=H7L52M2O0O010O1001N4M3M2M4lNmF]OU9`0lF@W9<kFCX98jFHY94iFJ[92fFN]9NeF1^9JdF6_9FcF9`9CaF=b9^OaFa0b9[O`Fc0\:O001O1O1O000001WE_OY:a0cEB^:>^EFb::[EHf:e00O1O100O1N2O1N2N3M2O1N2N2O2M2M3Ka^?'

# print(rle2bbox(rle, (375, 1242)))