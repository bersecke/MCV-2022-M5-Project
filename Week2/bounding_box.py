from pycocotools import mask
import numpy as np

with open('dataset/0000.txt','r') as fd:
    lines = fd.readlines()

line0 = lines[0].split(' ')
print('line0: ',line0[5])

# in_ = np.reshape(np.asfortranarray(line0[5]), (int(line0[3]), int(line0[4]), 1))
# in_ = np.asfortranarray(in_)
# rle = mask.encode(in_)

rle = mask.decode({'size': [int(line0[3]), int(line0[4])], 'counts': line0[5]})
print('rle:', rle)

bbox = mask.toBbox(rle).tolist()
print(box)

