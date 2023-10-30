# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 11:23:52 2023

@author: Administrator
"""

import cv2
import glob

im_path = "F://YaelShammaiData//data//images//train1//*.png"

filenames = sorted(glob.glob(im_path))

for f in filenames:
  img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
  width = 640
  height = 640.
  dim = (width, height)
   
# resize image
  resized = cv2.resize(img, (640, 640), interpolation = cv2.INTER_LINEAR)
  #find file name 
  
  aux= f.split('\\')
  outputf ="F://YaelShammaiData//data//images//resize//" + aux[len(aux)-1]
  print(outputf)
  cv2.imwrite(outputf, resized)
  
cv2.waitKey(0)
cv2.destroyAllWindows()



