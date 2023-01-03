# -*- coding: utf-8 -*-
import numpy as np 
import cv2 
import os

folder_path="../../calibration"
files_list = np.array(os.listdir(folder_path))

image_file_list=[]
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


cal_list=[]
cal_list2=[]

objp = np.zeros((6*8,3), np.float32)
objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)
objp = objp * (25 / 9)


for i in range(len(files_list)):
    if files_list[i][-4:] == '.png':
        image_file_list.append(files_list[i])

for i in range(len(image_file_list)):
    img = cv2.imread(folder_path + "/" + image_file_list[i])
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    val, corners = cv2.findChessboardCorners(gray_image, (8,6), None)
    
    if val==True:
        cal_list.append(objp)
        corners2 = cv2.cornerSubPix(gray_image, corners, (11,11),(-1,-1), criteria)
        cal_list2.append(corners2)
        
        cv2.drawChessboardCorners(img, (8,6), corners2, val)
        cv2.imshow('img', img)
        cv2.waitKey(200)
        
        
#cv2.destroyAllWindows()


ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(cal_list, cal_list2, gray_image.shape[::-1], None, None)

np.save("calibration_maxtix.npy",mtx)
np.save("calibration_dist.npy",dist)
#img = cv.imread('left12.jpg')
h,  w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

# undistort
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('calibresult.png', dst)