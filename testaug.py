# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 09:35:20 2023

@author: Administrator
"""
import albumentations as A
import cv2



###########
bboxes = []
new_bboxes = []
text_file = "F://PoseYolo//train//data//labels//train//img0001.txt"

f = open(text_file, "r")
 #read each file
line_all = f.read()
line_all = line_all.split('\n')
for line in line_all:
 #convert into list 
    list_aux = line.split(' ')
 #convert into a list order as should be
    list_aux_1 = [float(list_aux[1]),float(list_aux[2]),float(list_aux[3]),float(list_aux[4]),list_aux[0]]
 # do the same with two lines
 #append to another listlist_aux[1]
    bboxes.append(list_aux_1)



###############3

#transform = A.Compose([A.HorizontalFlip(p=0.5)])
transform = A.Compose([A.VerticalFlip(p=0.5)],bbox_params=A.BboxParams(format='yolo'))
image = cv2.imread("F://PoseYolo//train//data//images//train//img0001.png")
transformed = transform(image=image,bboxes = bboxes)
transformed_image = transformed["image"]
transformed_bboxes = transformed['bboxes']
###
count=0
string_list = []
file1= open("F://PoseYolo//train//data//augmentated//labels//bimg0001.txt","w")
for l in transformed_bboxes:
    laux = [l[0],l[1],l[2],l[3]]
    laux_string = ' '.join([str(item) for item in laux])
    laux_string = str(count) + ' ' + laux_string + "\n"
    string_list.append(laux_string)
    count = count + 1
file1.writelines(string_list)
file1.close()

cv2.imwrite("F://PoseYolo//train//data//augmentated//images//bimg0001.png",transformed_image)



