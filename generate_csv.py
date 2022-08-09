import csv
import os
arr = os.listdir('ETIS-LaribPolypDB/labels')
print(arr[0].split('.')[0])
list1=[]
for i in range(len(arr)):
    list1.append(arr[i].split('.')[0])
list1.sort()
for i in range(len(arr)):
    list1[i]=list1[i]+'.txt'
    #print(list1[i])


import os
arr1 = os.listdir('ETIS-LaribPolypDB/images')
print(arr1[0])


list2=[]
for i in range(len(arr)):
    list2.append(arr[i].split('.')[0])
list2.sort()
print(list2)
for i in range(len(arr)):
    list2[i]=list2[i]+'.jpg'
    print(list2[i], list1[i])




with open("etis.csv", mode="w", newline="") as train_file:
    for line in range(len(arr)):
        # print(line)
        # image_file = line.split("/")[-1].replace("\n", "")
        # text_file = image_file.replace(".jpg", ".txt")
        data = [list2[line], list1[line]]
        writer = csv.writer(train_file)
        writer.writerow(data)






































# import os
# import csv
# import os
# arr = os.listdir('generate-labels')
# print(arr[0].split('.')[0])
# list1=[]
# for i in range(len(arr)):
#     list1.append(arr[i].split('.')[0])
# list1.sort()
# for i in range(len(arr)):
#     list1[i]=list1[i]+'.txt'
#     #print(list1[i])


# import os
# arr1 = os.listdir('images')
# print(arr1[0])


# list2=[]
# for i in range(len(arr)):
#     list2.append(arr[i].split('.')[0])
# list2.sort()
# print(list2)
# for i in range(len(arr)):
#     list2[i]=list2[i]+'.jpg'
#     print(list2[i], list1[i])




# with open("train.csv", mode="w", newline="") as train_file:
#     for line in range(len(arr)):
#         # print(line)
#         # image_file = line.split("/")[-1].replace("\n", "")
#         # text_file = image_file.replace(".jpg", ".txt")
#         data = [list2[line], list1[line]]
#         writer = csv.writer(train_file)
#         writer.writerow(data)





# read_train = open("train.txt", "r").readlines()

# with open("train.csv", mode="w", newline="") as train_file:
#     for line in read_train:
#         image_file = line.split("/")[-1].replace("\n", "")
#         text_file = image_file.replace(".jpg", ".txt")
#         data = [image_file, text_file]
#         writer = csv.writer(train_file)
#         writer.writerow(data)

# read_train = open("test.txt", "r").readlines()

# with open("test.csv", mode="w", newline="") as train_file:
#     for line in read_train:
#         image_file = line.split("/")[-1].replace("\n", "")
#         text_file = image_file.replace(".jpg", ".txt")
#         data = [image_file, text_file]
#         writer = csv.writer(train_file)
#         writer.writerow(data)
