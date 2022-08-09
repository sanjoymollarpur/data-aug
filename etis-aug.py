import json
import csv
import cv2
import albumentations as A
import numpy as np
from utils import plot_examples
from PIL import Image
from types import SimpleNamespace
# p = "kavsir_bboxes.json"
# f = open(p)
#data = json.loads(f, object_hook=lambda d: SimpleNamespace(**d))
# data = json.load(f)
file = open('etis.csv')
csvreader = csv.reader(file)
header = []
header = next(csvreader)

for i in range(196):
    header = next(csvreader)
    #print(header[0], header[0].split('.')[0])
    label_path = "ETIS-LaribPolypDB/labels/"+header[1]
    boxes = []
    k=i
    image_list=[]
    with open(label_path) as f:
        for label in f.readlines():
        
            class_label, x, y, width, height = [
                float(x) if float(x) != int(float(x)) else int(x)
                for x in label.replace("\n", "").split()
            ]
            
            boxes.append([x, y, width, height])

    print(boxes)
    # print(header[0])
    image = cv2.imread("ETIS-LaribPolypDB/bgr-img/"+header[0])
    # image = Image.open("cvc-clinicDB/bgr-img/"+header[0])
    # cv2.imshow("img", image)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image=np.array(image)
    bboxes = [[38, 5, 430, 338]]
    SIZE=448
    saved_bboxes = []
    image_list.append(image)
    #plot_examples(image_list,boxes)
    # Pascal_voc (x_min, y_min, x_max, y_max), YOLO, COCO

    transform = A.Compose(
        [
            A.Resize(width=SIZE, height=SIZE),
            
        ],bbox_params=A.BboxParams(format="pascal_voc", min_area=448,
                                    min_visibility=0.3, label_fields=[])
    )

    images_list = [image]
    saved_bboxes = [boxes]

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
    # print(boxes[0])
    # print(convert(size, boxes[0]))
    print("images...")
    for i in range(2):
        augmentations = transform(image=image, bboxes=boxes)
        augmented_img = augmentations["image"]

        if len(augmentations["bboxes"]) == 0:
            continue

        images_list.append(augmented_img)
       # print(augmentations["bboxes"][0])
        box2 = augmentations["bboxes"]
        size=[SIZE, SIZE]
        box1=[]
        # print(box2)
        for j in box2:
            box1.append(convert(size, j))
        #print("after resize", convert(size, box))
        # box1=convert(size, box)
        #print(box1)
        saved_bboxes.append(augmentations["bboxes"])
        with open(f"etis-aug-labels/{header[0].split('.')[0]}.txt", 'w') as f:
            list = []
            for b1 in box1:
                listItem = f"0 {b1[0]} {b1[1]} {b1[2]} {b1[3]}\n"
                
                list.append(listItem) 
            f.writelines(list)
        #img = Image.open("aug-img/"+{i}+{header[0].split('.')[0].jpg})
        #print(augmented_img.shape)
        filename=f"etis-aug-img/{header[0].split('.')[0]}.jpg"
        #print(filename)
        augmented_img=cv2.cvtColor(augmented_img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(filename, augmented_img) 
    print(saved_bboxes)

    # plot_examples(images_list, saved_bboxes,k)




# import json
# import csv
# import cv2
# import albumentations as A
# import numpy as np
# from utils import plot_examples
# from PIL import Image
# from types import SimpleNamespace
# p = "kavsir_bboxes.json"
# f = open(p)
# #data = json.loads(f, object_hook=lambda d: SimpleNamespace(**d))
# data = json.load(f)
# file = open('8examples.csv')
# csvreader = csv.reader(file)
# header = []
# header = next(csvreader)

# for i in range(16):
#     header = next(csvreader)
#     print(header[0], header[0].split('.')[0])
#     label_path = "generate-labels/"+header[1]
#     boxes = []
#     image_list=[]
#     with open(label_path) as f:
#         for label in f.readlines():
        
#             class_label, x, y, width, height = [
#                 float(x) if float(x) != int(float(x)) else int(x)
#                 for x in label.replace("\n", "").split()
#             ]
            
#             boxes.append([x, y, width, height])

#     print(boxes)
#     image = cv2.imread("images/"+header[0])
#     #mask = Image.open("masks/cju0qkwl35piu0993l0dewei2.jpg")
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     bboxes = [[38, 5, 430, 338]]
#     SIZE=448
#     saved_bboxes = [bboxes[0]]
#     #image_list.append(image)
#     #plot_examples(image_list,boxes)
#     # Pascal_voc (x_min, y_min, x_max, y_max), YOLO, COCO

#     transform = A.Compose(
#         [
#             A.Resize(width=SIZE, height=SIZE),
#             A.RandomCrop(width=440, height=440),
#             A.Rotate(limit=40, p=0.9, border_mode=cv2.BORDER_CONSTANT),
#             A.HorizontalFlip(p=0.5),
#             A.VerticalFlip(p=0.1),
#             A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.9),
#             A.OneOf([
#                 A.Blur(blur_limit=3, p=0.5),
#                 A.ColorJitter(p=0.5),
#             ], p=1.0),
#         ], bbox_params=A.BboxParams(format="pascal_voc", min_area=448,
#                                     min_visibility=0.3, label_fields=[])
#     )

#     images_list = [image]
#     saved_bboxes = [boxes[0]]

#     def convert(size, box):
#         dw = 1./size[0]
#         dh = 1./size[1]
#         x = (box[0] + box[2])/2.0
#         y = (box[1] + box[3])/2.0
#         w = box[2] - box[0]
#         h = box[3] - box[1]
#         x = x*dw
#         w = w*dw
#         y = y*dh
#         h = h*dh
#         return (x,y,w,h)
#     size=[SIZE, SIZE]
#     print(boxes[0])
#     print(convert(size, boxes[0]))
#     print("images .....\n\n\n")
#     for i in range(5):
#         augmentations = transform(image=image, bboxes=boxes)
#         augmented_img = augmentations["image"]

#         if len(augmentations["bboxes"]) == 0:
#             continue

#         images_list.append(augmented_img)
#         print(augmentations["bboxes"][0])
#         box = augmentations["bboxes"][0]
#         size=[SIZE, SIZE]
#         print("after resize", convert(size, box))
#         saved_bboxes.append(augmentations["bboxes"][0])

#     plot_examples(images_list, saved_bboxes)
