# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 10:26:25 2023

@author: Administrator
"""
import numpy as np
import pandas as pd
import glob
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
        #create an empty dataframe
        self.df = pd.DataFrame(columns = ['frame', 'x-center-rec','y-center-rec','width-rec','height-rec'])
        self.dataframepoints = pd.DataFrame()
        self.Total = pd.DataFrame()
        
        
    '''
    Create rectangle dataframe
    '''

    def GetRectangleInf(self,Data):
        
         while len(Data) > 0:
             #Remove first element of the list
             
             B = Data.pop(0)
             
             
             width = abs(B[0,1]-B[2,1])
             height = abs(B[0,1]-B[0,2])
             xcenter = (B[0,1]+B[2,1])/2
             ycenter = (B[0,2]+B[2,2])/2
            
             self.df.loc[len(self.df)] = {'frame' : B[0,0], 'x-center-rec' : ycenter, 'y-center-rec' : xcenter, 'width-rec' : height, 'height-rec' : width}
       
         
         self.df.to_csv('test.csv', sep=',', index=False, encoding='utf-8')
        
    '''
    Create data frame for points
    '''
         
    def GetPointsInf(self,labels,points,LABELS):
        #convert numpy array into pandas array
        labelsframe= pd.DataFrame(labels['label'])
        
        pointsframe = pd.DataFrame(points)
       
        aux_1 = pd.concat([pointsframe,labelsframe], axis = 1)
        aux_1.columns = ['0','1','2','3']
        aux_2 = aux_1.pivot(index = '0',columns = '3')
        
       
        columnsnumber = auxiliary(LABELS)
        
        self.dataframepoints = aux_2.iloc[:,columnsnumber]
        
        #in order to get the correct xy reorder the data
        self.dataframepoints = OrderData(self.dataframepoints)
        
    '''
    Join points with planes
    '''    
    def FusionData(self,PATH_FOLDER):
            self.Total = pd.concat([self.df,self.dataframepoints], axis = 1)

            path = PATH_FOLDER + '//test2.csv'
            self.Total.to_csv(path, sep=',', index=False, encoding='utf-8')
        
        
        
    '''
    Conversion pandas to text 
    '''
    def ConverionPandastoText(self,PATH_FOLDER,shape):
        #get all the files
        output = GetFilenames(PATH_FOLDER)
        for index in range(len(self.Total.index)):
            any_row = self.Total.iloc[index,self.Total.columns!='frame']
            #normalize the data
            any_row = Normalize(any_row,shape)
            
            #Get any row of dataframe as string
            any_row_as_string = any_row.to_string(index=False)
            #Replace '\n' with comma
            any_row_as_string = '0'+' '+ any_row_as_string.replace('\n', ' ')
            print(any_row_as_string)
            #save as txt file
            file1 = open(output[index],"w")#write mode
            file1.write(any_row_as_string)
            file1.close()
    
'''

Auxiliary functions
'''

def auxiliary(LABELS):
        columnsnumber = []
        #create vector with numbers
        for index  in  range(0, len(LABELS)):
            columnsnumber.append(index)
            columnsnumber.append(index+len(LABELS))
        return columnsnumber
    
'''
#get all the files for text
input: folder name
output: list of filenames
'''

def GetFilenames(PATH_FOLDER):
    output = []
    #get all filenames of the images
    im_path = PATH_FOLDER + '//images//train//*.png'
    
    filenames = sorted(glob.glob(im_path))
    
    for f in filenames:
       aux = PATH_FOLDER + '//labels//train//' + (((f.split('\\'))[1]).split('.'))[0] + '.txt'
       output.append(aux)
       
    return output
    
'''
Normalize the data with the width and height
input : data, w and h
output: return the normalized data
'''
def  Normalize(any_row,shape):
    
    for index in range(len(any_row)):
        if index % 2 == 0:
           any_row[index] = any_row[index]/shape[2]
        else:
           any_row[index] = any_row[index]/shape[1]
    return any_row

'''
Reorder the data in order to get correct x and y
'''
def OrderData(data):
    
   new = []
   for index in range(0,len(data.columns),2):
      new.append(index+1)
      new.append(index)
     
   data_new = data.iloc[:,new]
   
   return data_new