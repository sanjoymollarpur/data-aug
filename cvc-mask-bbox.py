import os
import numpy as np
import cv2
import glob
from tqdm import tqdm
from skimage.measure import label, regionprops, find_contours
from matplotlib import pyplot as plt
from PIL import Image


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


# bboxes = mask_to_bbox(y)

SIZE=448

def convert(size, box):
        dw = 1./size[0]
        dh = 1./size[1]
        x = (box[0] + box[2])/2.0
        y = (box[1] + box[3])/2.0
        w = box[2] - box[0]
        h = box[3] - box[1]
        x = x*dw
        w = w*dw
        y = y*dh
        h = h*dh
        return (x,y,w,h)
size=[SIZE, SIZE]


for  i in glob.glob("cvc-clinicDB/PNG/GroundTruth/*.png"):
    y = cv2.imread(i,0)
    path=(i.split("/")[3]).split(".")[0]
    print(path)
    x = cv2.imread(f'cvc-clinicDB/PNG/Original/{path}.png')
    # x = Image.fromarray(x)
    # x=np.array(x)
    # x = x.resize((SIZE, SIZE))
    # x=np.array(x)
    img_path=f'cvc-clinicDB/PNG/bgr-img/{path}.png'
    cv2.imwrite(img_path, x)
    print(x.shape, y.shape)
    SIZE1=x.shape[0]
    SIZE2=x.shape[1]
    size=[SIZE1, SIZE2]
    # x = Image.fromarray(x)
    # x = x.resize((SIZE, SIZE))
    # print(x)
    box1 = mask_to_bbox(y)
    box2=[]
    for p in box1:
      box2.append(convert(size, p))
    print(box1[0])
    print(x.shape, y.shape)
    with open(f"cvc-clinicDB/PNG/labels/{path}.txt", 'w') as f:
        list = []
        for k in box2:
            listItem = f"0 {k[0]} {k[1]} {k[2]} {k[3]}\n"
            list.append(listItem) 
        f.writelines(list)



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
    # plt.show()


# upper_left_x = box[0] - box[2] / 2
# upper_left_y = box[1] - box[3] / 2
# rect = patches.Rectangle(
#     (upper_left_x * width, upper_left_y * height),
#     box[2] * width,
#     box[3] * height,
#     linewidth=2,
#     edgecolor="red",
#     facecolor="none",
# )

# # Add the patch to the Axes
# t = confi
# ax.add_patch(rect)
# plt.text(
#     upper_left_x * width,
#     upper_left_y * height,
#     s=t,
#     color="white",
#     verticalalignment="top",
#     bbox={"color": colors[int(class_pred)], "pad": 0},
# )