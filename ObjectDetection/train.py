# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 09:28:43 2023

@author: Administrator
"""
import cv2
from ultralytics import YOLO
import torch



import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# #Set up GPU
# device = "0" if torch.cuda.is_available() else "cpu"
# if device == "0":
#    torch.cuda.set_device(0)
# print("Device:",device)

# Load  a pretained  YOLOv8 model
#model = YOLO('yolov8l.pt')
model = YOLO('yolov8l-pose.pt')

#model.train(data='F:/YaelShammaiData/Roboflow/conf.yaml', epochs=500, name = 'yoloRNew',workers = 0,patience=0)  # train the model
#model.train(data='F:/YaelShammaiData/data/conf_box_1.yaml', epochs=500, name = 'yoloG', workers = 0,patience=0)  # train the model
#model.train(data='F:/YaelShammaiData/dataPose/conf_pose.yaml', epochs=500, name = 'yoloPoseA', workers = 0,patience=0)  # train the model
model.train(data='F:/YaelShammaiData/dataPoseWhite/conf_pose.yaml', epochs=500, name = 'yoloPoseWhite', workers = 0,patience=0)  # train the model