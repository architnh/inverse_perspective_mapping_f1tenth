# -*- coding: utf-8 -*-
import numpy as np 
import cv2 
import os
import matplotlib.pyplot as plt

class distances:
    
    def get_car_coords(self, x_in, y_in):
        x_car = (self.fy / (y_in-self.y0)) * (self.z_car + self.h_mount)
        y_car = (x_in - self.x0) * (x_car / self.fx)
        return x_car, y_car
        
    def __init__(self):  
        folder_path= "../../distance" #"../imgs"#"../distance"
        files_list = np.array(os.listdir(folder_path))
        
        mtx = np.load("calibration_maxtix.npy")
        dist = np.load("calibration_dist.npy")
        
        
        
        #Calibrate Images
        image_file_list=[]
        for i in range(len(files_list)):
            if files_list[i][-4:] == '.png':
                image_file_list.append(files_list[i])
          
        calibrated_images=[]
        for i in range(len(image_file_list)):
            img = cv2.imread(folder_path + "/" + image_file_list[i])
          
            h,  w = img.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
            
            # undistort
            dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
            # crop the image
            x, y, w, h = roi
            calibrated_images.append(dst[y:y+h, x:x+w])
            #cv2.imshow(str(i), dst[y:y+h, x:x+w])
            plt.figure(i)
            plt.imshow(img)#dst[y:y+h, x:x+w])
            #Finding Mount Height using provided distance image
            self.y0 = 0
            self.x0 = 0
            self.x_car =0.4#Distance in meters of car from cone
            self.z_car = 0#Height of object above the ground
            self.fx = mtx[0,0]
            self.fy = mtx[1,1]
            y = 496.95
            if i==0:
                self.shape=img.shape
            #y =  img.shape[0] 
            print("image shape=",img.shape)
            self.h_mount = (y - self.y0) * (self.x_car / self.fy) - self.z_car
        
        
        
        print("Mount Height (cm) =",self.h_mount * 100)
    
        #Get distance to cone in distance folder
        x_val,y_val = self.get_car_coords(597.99,415.82)
        print("x_distance(cm) =",x_val*100)
        print("y distance(cm)",y_val*100)
        
        
         #Get distance to cone_point in imgs folder
        x_val,y_val = self.get_car_coords(284.50,331.38)
        print("Distance to cone with pointer")
        print("x_distance(cm) =",x_val*100)
        print("y distance(cm)",y_val*100)
      
        
      
        
distances()       
        