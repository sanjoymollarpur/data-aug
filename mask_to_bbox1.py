import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
from skimage.measure import label, regionprops, find_contours
from matplotlib import pyplot as plt
from PIL import Image



y = cv2.imread('cvc-clinicDB/PNG/GroundTruth/5.png',0)
x = cv2.imread('cvc-clinicDB/PNG/Original/5.png', cv2.IMREAD_COLOR)

print(x.shape, y.shape)
SIZE = 256
# x = Image.fromarray(x)
# x = x.resize((SIZE, SIZE))
# print(x)

print(x.shape, y.shape)

plt.figure(figsize=(16, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(x)
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(y)
# plt.subplot(233)
# plt.title('Prediction on test image')
# plt.imshow(prediction, cmap='gray')
plt.show()


def mask_to_border(mask):
    h, w = mask.shape
    border = np.zeros((h, w))

    contours = find_contours(mask, 128)
    for contour in contours:
        for c in contour:
            x = int(c[0])
            y = int(c[1])
            border[x][y] = 255

    return border


def mask_to_bbox(mask):
    bboxes = []

    mask = mask_to_border(mask)
    lbl = label(mask)
    props = regionprops(lbl)
    for prop in props:
        x1 = prop.bbox[1]
        y1 = prop.bbox[0]

        x2 = prop.bbox[3]
        y2 = prop.bbox[2]

        bboxes.append([x1, y1, x2, y2])

    return bboxes

def parse_mask(mask):
    mask = np.expand_dims(mask, axis=-1)
    mask = np.concatenate([mask, mask, mask], axis=-1)
    return mask


bboxes = mask_to_bbox(y)

    
for bbox in bboxes:
    x = cv2.rectangle(x, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)

cat_image = np.concatenate([x, parse_mask(y)], axis=1)



plt.figure(figsize=(16, 8))
plt.subplot(232)
plt.title('Testing Image')
plt.imshow(x)
# plt.subplot(232)
# plt.title('Testing Label')
# plt.imshow(y)
plt.subplot(231)
plt.title('Prediction on test image')
plt.imshow(cat_image)
plt.show()
