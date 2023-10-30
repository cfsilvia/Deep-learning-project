# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 12:40:29 2023

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 10:26:25 2023

@author: Administrator
"""
import numpy as np
import pandas as pd
import glob
from dask_image.imread import imread
'''
This library treats the data from the napari to give the correct format
'''

'''
function to read the data from the rect for each image
Input: list of arrays. Each array save four points for each frame,
Ouput: pandas frame each row has number of frame x,y center rect and width large of the rectangle 
'''
class HelperFunctions:
    def __init__(self):
        # create an empty dataframe
        self.df = pd.DataFrame(columns = ['frame', 'x-center-rec','y-center-rec','width-rec','height-rec', 'filename'])
       
        
        self.Total = pd.DataFrame(columns = ['frame', 'x-center-rec','y-center-rec','width-rec','height-rec'])
        
        
    '''
    Create rectangle dataframe
    '''

    def GetRectangleInf(self,Data,mouse,im_path,PATH_FOLDER ):
         #print(Data)
         
         #real size of the pictures
         size_list = getSize(im_path)
         
         count  = 0
         filenames = sorted(glob.glob(im_path))
         while len(Data) > 0:
             #Remove first element of the list
             
             B = Data.pop(0)
             sizeP = size_list.pop(0)
            
         
             height = abs(B[3,1]-B[0,1])
             width = abs(B[2,2]-B[3,2])
             ycenter = (B[0,1]+B[2,1])/2
             xcenter = (B[0,2]+B[2,2])/2
            
             self.df.loc[len(self.df)] = {'frame' : B[0,0], 'x-center-rec' : xcenter, 'y-center-rec' : ycenter, 'width-rec' : width, 'height-rec' : height, 'filename':filenames[count]}
             #normalize[]
             self.Total.loc[len(self.Total)] = {'frame' : B[0,0], 'x-center-rec' : xcenter/sizeP[2], 'y-center-rec' : ycenter/sizeP[1], 'width-rec' : width/sizeP[2], 'height-rec' : height/sizeP[1]}
             count += 1
             
         filename = PATH_FOLDER  + '//' + mouse + '.csv'
         print(filename)
         self.df.to_csv(filename, sep=',', index=False, encoding='utf-8')
         return self.Total   
     
        
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
