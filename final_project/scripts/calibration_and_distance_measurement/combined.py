# -*- coding: utf-8 -*-
import numpy as np 
import cv2 
import os
import matplotlib.pyplot as plt

import torch
import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import cv2
import torch.nn as nn

#Height of Realsense on the car
height = 11.43#cm


vid = cv2.VideoCapture(0)


  
while(True):
      
    ret, frame = vid.read()
  
    
    #Lane Detections
    blur_image = cv2.GaussianBlur(frame, (5,5), 2)
    
    '''
    plt.figure(1)
    plt.imshow(blur_image, animated=True)
    plt.draw()
    plt.pause(0.0001)
    plt.clf()'''
    
    hsv = cv2.cvtColor(blur_image, cv2.COLOR_BGR2HSV)
    
    '''
    plt.figure(2)
    plt.imshow(hsv,animated=True)
    plt.draw()
    plt.pause(0.0001)
    plt.clf()'''
    
    mask = cv2.inRange(hsv, np.array([20, 70, 70]), np.array([40, 255, 255]))
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(hsv,hsv, mask= mask)
    '''
    plt.figure(3)
    plt.imshow(res,animated=True)
    plt.draw()
    plt.pause(0.0001)
    plt.clf()'''

    
    contours, hierarchy = cv2.findContours(image=mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    # draw contours on the original image
    image_copy = frame.copy()
    cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=5, lineType=cv2.LINE_AA)
    
    
    # Display the resulting frame
    cv2.imshow('frame', image_copy)
      

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()