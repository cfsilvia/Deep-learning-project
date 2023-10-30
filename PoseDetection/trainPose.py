# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 09:43:59 2023

@author: Administrator
"""

import cv2
from ultralytics import YOLO


# Load  a pretained  YOLOv8 model
model = YOLO('yolov8m-pose.pt')


model.train(data='F:/PoseYolo/train/data/conf.yaml', epochs=50, name = 'yoloSecondTestm1')  # train the model
