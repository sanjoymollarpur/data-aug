import random
import cv2
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import numpy as np
import albumentations as A


def visualize(image):
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(image)
    plt.show()


def plot_examples(images, bboxes=None, j=0):
    fig = plt.figure(figsize=(25, 25))
    columns = 6
    rows = 6
    print("bboxes", bboxes)
    for i in range(0, 2):
    # for i in range(0, len(images)):
        if bboxes is not None:
            for k in range(len(bboxes[i])):
                img = visualize_bbox(images[i], bboxes[i][k], class_name="Elon")
            print("plot \n",bboxes[i][0])
        else:
            img = images[i]
        fig.add_subplot(rows, columns, i+1)
        plt.imshow(img)
    # plt.savefig(f"aug-frame/{j}.jpg")
    plt.show()


def visualize_bbox(img, bbox, class_name, color=(255, 0, 0), thickness=5):
    """Visualizes a single bounding box on the image"""
    # print("visu", bbox)
    x_min, y_min, x_max, y_max =bbox
    x_min, y_min, x_max, y_max =int(x_min), int(y_min), int(x_max), int(y_max)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, thickness)
    return img

