# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 12:00:08 2023

@author: Administrator
"""
import albumentations as A
import cv2

'''
Group of function which help to augment the pictures and to save in the correct format
'''

class AugmentationFunctions:
    def __init__(self,filename,path_folder):
       self._filename = filename
       self._path = path_folder
       #read image
       self._image = cv2.imread(filename) 
       self.total_list = []
       self._name = ' '
    
    def arrangeBbox(self):
        
        #arrange path
        aux_1 = self._filename.split('\\')
        aux_2 = aux_1[len(aux_1)-1].split('.')
        text_file = self._path + '//labels//train//'+ aux_2[0] + '.txt'
        self._name = aux_2[0]
        #read text box
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
           self.total_list.append(list_aux_1)
        
           
        f.close()

    def augmentation(self):
        bboxes = self.total_list
        image = self._image
       ###############3
        
        transform = A.Compose([A.HorizontalFlip(p=0.5)],bbox_params=A.BboxParams(format='yolo'))
        #transform = A.Compose([A.VerticalFlip(p=0.5)],bbox_params=A.BboxParams(format='yolo'))
       
        transformed = transform(image=image,bboxes = bboxes)
        transformed_image = transformed["image"]
        transformed_bboxes = transformed['bboxes']
       ###
       
        save_pictures = self._path + '//augmentated//images//'+ 'fh_' + self._name + '.png'
        save_text = self._path + '//augmentated//labels//'+ 'fh_' + self._name + '.txt'
        print(save_pictures)
        #save picture
        cv2.imwrite(save_pictures, transformed_image)
        
        #save text
        count=0
        string_list = []
        file1= open(save_text,"w")
        for l in transformed_bboxes:
            laux = [l[0],l[1],l[2],l[3]]
            laux_string = ' '.join([str(item) for item in laux])
            laux_string = str(count) + ' ' + laux_string + "\n"
            string_list.append(laux_string)
            count = count + 1
        file1.writelines(string_list)
        file1.close()
        
    def augmentationImageHorizontal(self):
            #arrange path
            aux_1 = self._filename.split('\\')
            aux_2 = aux_1[len(aux_1)-1].split('.')
            self._name = aux_2[0]
            image = self._image
           ###############3
            
            transform = A.Compose([A.HorizontalFlip(p=0.5)])
            #transform = A.Compose([A.VerticalFlip(p=0.5)],bbox_params=A.BboxParams(format='yolo'))
           
            transformed = transform(image=image)
            transformed_image = transformed["image"]
           
           ###
           
            save_pictures = self._path + '//augmentated//images//'+ 'fh_' + self._name + '.png'
            print(save_pictures)
           
            #save picture
            cv2.imwrite(save_pictures, transformed_image)
            
    def augmentationImageVertical(self):
              
                image = self._image
               ###############3
                
                transform = A.Compose([A.VerticalFlip(p=0.5)])
                #transform = A.Compose([A.VerticalFlip(p=0.5)],bbox_params=A.BboxParams(format='yolo'))
               
                transformed = transform(image=image)
                transformed_image = transformed["image"]
               
               ###
               
                save_pictures = self._path + '//augmentated//images//'+ 'fv_' + self._name + '.png'
               
                #save picture
                cv2.imwrite(save_pictures, transformed_image)
            
            
#Auxiliary functions
def verify(list_data):
    
    index = 0
    for index in range(4):
        if  list_data[index] < 0:
          list_data[index] = 0
        index = index +1
    return list_data