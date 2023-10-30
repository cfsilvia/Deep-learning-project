# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 13:48:48 2023

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
import RearrangeData 
import pathlib
from tkinter import filedialog
import tkinter as tk
import ObjectDetection
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


# COLOR_CYCLE = [
#     '#1f77b4',
#     '#ff7f0e',
#     '#2ca02c'




def create_label_menu(points_layer, labels):
    """Create a label menu widget that can be added to the napari viewer dock

    Parameters:
    -----------
    points_layer    napari.layers.Points
        a napari points layer
    labels : List[str]
        list of the labels for each keypoint to be annotated (e.g., the body parts to be labeled).

    Returns:
    --------
    label_menu : Container
        the magicgui Container with our dropdown menu widget
    """
    # Create the label selection menu
    label_menu = ComboBox(label='feature_label', choices=labels)
    label_widget = Container(widgets=[label_menu])
###########################
##################################


##################################################################

    def update_label_menu(event):
        """Update the label menu when the point selection changes"""
        new_label = str(points_layer.current_properties['label'][0])
        if new_label != label_menu.value:
            label_menu.value = new_label
        print('updated')
        print(new_label)
        print(points_layer.data)
      
        

    points_layer.events.current_properties.connect(update_label_menu)

    def label_changed(event):
        """Update the Points layer when the label menu selection changes"""
        selected_label = event.value
        current_properties = points_layer.current_properties
        current_properties['label'] = np.asarray([selected_label])
        points_layer.current_properties = current_properties
        print('changed')

    label_menu.changed.connect(label_changed)

    return label_widget

def create_viewer( im_path: str,path: str):
     global SHAPE_STACK
     global IM_PATH
     global PATH_FOLDER
     PATH_FOLDER = path
     IM_PATH = im_path
    #function to read the images
     stack = LoadImages(im_path)
    # LABELS = labels
    # stack = imread(im_path)
     SHAPE_STACK = stack.shape
     print(SHAPE_STACK)
     viewer = napari.view_image(stack)
     return viewer
    
def point_annotator(viewer,
       
        points_layer_name, color_name, shape_name,labels: List[str]):

    global LABELS

   # features = {'label': np.empty(3, dtype=int)}
    properties = {'label': labels}
    # add the points
    #create a big layer and cut it for the number of labels
    arrayModel = [[0,100,-1],[0,200,-1],[0,300,-1],[0,400,-1],[0,500,-1],[0,600,-1],[0,700,-1],[0,800,-1],[0,900,-1],[0,1000,-1]]
    points= arrayModel[0:len(labels)]
   # points = np.array([[0,100,-1],[0,200,-1],[0,300,-1],[0,400,-1]])
   
    # for index in range(1,SHAPE_STACK[0]):
    #     points =  np.append(points,arrayModel[0:len(labels)])       
        
    
    points_layer = viewer.add_points(
    points,
    properties=properties,
    edge_color='label',
    edge_color_cycle=COLOR_CYCLE,
    symbol='o',
    face_color = 'label',
    face_color_cycle=COLOR_CYCLE,
    edge_width_is_relative = 8,
    name = points_layer_name,
    size = 30,
    ndim=3
    )
    points_layer.edge_color_mode = 'cycle'
    points_layer.face_color_mode = 'cycle'
   
#####################
# add the polygons
    polygons = []
    for index in range(SHAPE_STACK[0]):
        polygons.append(np.array([[ index,  879.52867728,  592.9339079 ],
           [   index        ,  879.52867728,  906.41696375],
           [   index        , 1260.79185332,  906.41696375],
           [   index        , 1260.79185332,  592.9339079 ]]))
    layer_shapes = viewer.add_shapes(
        polygons,
        shape_type='polygon',
        edge_width=3,
        edge_color=color_name,
        face_color='#0000',
        name = shape_name,
        )
    
    
    
####################################
  # add the label menu widget to the viewer
    label_widget = create_label_menu(points_layer, labels)
    viewer.window.add_dock_widget(label_widget)
    viewer.window.add_dock_widget(my_widget1,area='right')
 #viewer.window.add_dock_widget(widget2,area='right')
     

    @viewer.bind_key('.')
    def next_label(event=None):
        """Keybinding to advance to the next label with wraparound"""
        current_properties = points_layer.current_properties
        current_label = current_properties['label'][0]
        ind = list(labels).index(current_label)
        new_ind = (ind + 1) % len(labels)
        new_label = labels[new_ind]
        current_properties['label'] = np.array([new_label])
        points_layer.current_properties = current_properties

    def next_on_click(layer, event):
        """Mouse click binding to advance the label when a point is added"""
        if layer.mode == 'add':
            next_label()

            # by default, napari selects the point that was just added
            # disable that behavior, as the highlight gets in the way
            layer.selected_data = {}

    points_layer.mode = 'add'
    points_layer.mouse_drag_callbacks.append(next_on_click)

    @viewer.bind_key(',')
    def prev_label(event):
        """Keybinding to decrement to the previous label with wraparound"""
        current_properties = points_layer.current_properties
        current_label = current_properties['label'][0]
        ind = list(labels).index(current_label)
        n_labels = len(labels)
        new_ind = ((ind - 1) + n_labels) % n_labels
        new_label = labels[new_ind]
        current_properties['label'] = np.array([new_label])
        points_layer.current_properties = current_properties
     
    @viewer.bind_key('s')
    def save_data(event=None):
         print(points_layer.data)
         
   
#Auxiliary functions
def arrange(shape,labels,points,rect,PATH_FOLDER):
    print(shape)
    l = labels 
    p = points
    r = rect
    file = widget2()
    print(file)
    print(points)
   
    objectmouse = RearrangeData.HelperFunctions()
    objectmouse.GetRectangleInf(r,'mouse_0',IM_PATH,PATH_FOLDER)
    print('second')
    objectmouse.GetPointsInf(labels,points,LABELS)
    print('third')
    objectmouse.FusionData(PATH_FOLDER)
    objectmouse.ConverionPandastoText(PATH_FOLDER,shape,IM_PATH)
   
    a = 1
    


   
    
        
@magicgui(call_button='Save Data')
def my_widget1(layer: napari.layers.Points,array:ImageData,layerShape:napari.layers.Shapes):
#def my_widget1():
       # rect = array
       shape = SHAPE_STACK
       labels = layer.properties
        
       points = layer.data
       rect = layerShape.data
       
       arrange(shape,labels,points,rect,PATH_FOLDER)
      
       return 0
   
    
   
@magicgui(path={'mode': 'd'}, call_button='Run')
def widget2(path =  pathlib.Path.home()):
    print(path)
    return (path)

@magicgui(call_button='Augment the data')
def widget3( ):
    print(PATH_FOLDER)
    
#Load the pictures as a stack
def LoadImages(im_path):
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
    
    return stack