# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 08:53:33 2023

@author: Administrator
"""
from typing import List
from napari import Viewer
from dask_image.imread import imread
import napari
from magicgui.widgets import ComboBox, Container
import numpy as np
from magicgui import magicgui
from napari.types import ImageData
import glob
import pandas as pd
import RearrangeData_ObjectDetection 
import pathlib
from tkinter import filedialog
import tkinter as tk
from typing import List
import RearrangeData
import AugmentationFunctions
from skimage.io import imread

from dask import delayed
import dask.array as da


COLOR_CYCLE = [
    '#1f77b4',
    '#ff7f0e',
    '#2ca02c',
    '#d62728',
    '#9467bd',
    '#8c564b',
    '#e377c2',
    '#7f7f7f',
    '#bcbd22',
    '#17becf'
]

'''
Add boxes
'''
def box_annotator( im_path: str,path: str):
    
    global SHAPE_STACK
    global PATH_FOLDER
    global IM_PATH
    PATH_FOLDER = path
    IM_PATH = im_path
   # # global viewer   
   #  stack = imread(im_path)
   #  print(stack)
   
    filenames = sorted(glob.glob(im_path))
   # read the first file to get the shape and dtype
   # ASSUMES THAT ALL FILES SHARE THE SAME SHAPE/TYPE
    sample = imread(filenames[0])

    lazy_imread = delayed(imread)  # lazy reader
    lazy_arrays = [lazy_imread(fn) for fn in filenames]
    dask_arrays = [
    da.from_delayed(delayed_reader, shape=sample.shape, dtype=sample.dtype)
    for delayed_reader in lazy_arrays
    ]
# Stack into one large dask.array
    stack = da.stack(dask_arrays, axis=0)
   
   
   
   
    #get the size of each image
    size_pictures =  getSize(im_path)
   # print(size_pictures)
    SHAPE_STACK = stack.shape
    #print(SHAPE_STACK)

    viewer = napari.view_image(stack)
    #####################
    # add the polygons
    box_mouse0 = []
    box_mouse1 = []
    labels_shape = ['child','parent']
    properties = {'label':labels_shape}
        
       
    for index in range(SHAPE_STACK[0]):
      box_mouse0.append(np.array([[ index,  879.52867728,  592.9339079 ],
         [   index        ,  879.52867728,  906.41696375],
         [   index        , 1260.79185332,  906.41696375],
         [   index        , 1260.79185332,  592.9339079 ]]))
      box_mouse1.append(np.array([[ index,  879.52867728,  592.9339079 ],
         [   index        ,  879.52867728,  906.41696375],
         [   index        , 1260.79185332,  906.41696375],
         [   index       , 1260.79185332,  592.9339079 ]]))
      
    layer_shapes0 = viewer.add_shapes(
            box_mouse0,
            
            shape_type='rectangle',
            edge_width=3,
            edge_color = 'coral',
            
            face_color='#0000',
            name='mouse_0'
           
            )
    layer_shapes1 = viewer.add_shapes(
            box_mouse1,
            
            shape_type='rectangle',
            edge_width=3,
            edge_color = 'blue',
            
            face_color='#0000',
            name='mouse_1'
           
            )
      
      
        
    

  
    #################################### 


    viewer.window.add_dock_widget(my_widget_shape,area='right')
    viewer.window.add_dock_widget(widget3,area='right')
    viewer.window.add_dock_widget(widget4,area='right')
'''
Add menu for object detection
'''

@magicgui(call_button='Save Data of mouse_0 and mouse_1')
#def my_widget_shape(viewer: Viewer,layer_shapes1:napari.layer.Layer):
def my_widget_shape(viewer:Viewer):
       mouse_0 = viewer.layers[1]
       mouse_1 = viewer.layers[2]
       print(mouse_0.data)
       print(mouse_1.data)
       object0 = RearrangeData_ObjectDetection.HelperFunctions()
       object1 = RearrangeData_ObjectDetection.HelperFunctions()
       data_frame_0 = object0.GetRectangleInf(mouse_0.data,'mouse_0',IM_PATH,PATH_FOLDER )
       data_frame_1 = object1.GetRectangleInf(mouse_1.data,'mouse_1',IM_PATH,PATH_FOLDER )
       print(data_frame_0)
       ConverionTwoPandastoText(data_frame_0, data_frame_1)
       
       
       return 0

@magicgui(call_button='Augment the data')
def widget3( ):
    #get file with images
    filenames = sorted(glob.glob(IM_PATH))
    for f in filenames:
       print(f)
       object_augmentation = AugmentationFunctions.AugmentationFunctions(f,PATH_FOLDER)
       
       object_augmentation.arrangeBbox()
       object_augmentation.augmentation()

    
    
@magicgui(call_button='Augment images')
def widget4( ):
    #get file with images
    filenames = sorted(glob.glob(IM_PATH))
    for f in filenames:
       object_augmentation = AugmentationFunctions.AugmentationFunctions(f,PATH_FOLDER)
       #object_augmentation.arrangeBbox()
       object_augmentation.augmentationImageHorizontal()
       object_augmentation.augmentationImageVertical()



'''
Auxiliary functions


'''
'''
Get a list with the size of each picture
'''
def getSize(im_path):
    filenames = sorted(glob.glob(im_path))
    size_list = []
    for f in filenames:
        aux_s = imread(f)
        size_p =aux_s.shape
        size_list.append(size_p)
    return size_list

'''
Convert to text two data frames
'''
def ConverionTwoPandastoText(data_frame_0, data_frame_1):
    #get all the files
    output =RearrangeData.GetFilenames(PATH_FOLDER)
    #print(output)
    for index in range(len(data_frame_0.index)):
        any_row_0 = data_frame_0.iloc[index,data_frame_0.columns != 'frame']
        any_row_1 = data_frame_1.iloc[index,data_frame_1.columns != 'frame']
        #Get any row of dataframe as string
        list_0 = any_row_0.values.tolist()
        
        any_row_as_string_0 = ' '.join([str(item) for item in list_0])
        
        
        #print(any_row_as_string_0)
        #Replace '\n' with comma
        any_row_as_string_0a = '0'+' '+ any_row_as_string_0 + "\n"
        #Get any row of dataframe as string
        list_1 = any_row_1.values.tolist()
        any_row_as_string_1 = ' '.join([str(item) for item in list_1])
        #Replace '\n' with comma
        any_row_as_string_1a = '1'+' '+ any_row_as_string_1
        print(output[index])
        print(any_row_as_string_0a)
        file1 = open(output[index],"w")#write mode
        file1.writelines([any_row_as_string_0a,any_row_as_string_1a])
        file1.close()