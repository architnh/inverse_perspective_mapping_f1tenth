from time import time
import cv2
import time
# import torch
# import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
# import torch.nn as nn
# # import tensorrt as trt
# import pycuda.driver as cuda
# import pycuda.autoinit


# cap = cv2.VideoCapture("v4l2src device=/dev/video2 extra-controls=\"c,exposure_auto=3\" ! video/x-raw, width=960, height=540 ! videoconvert ! video/x-raw,format=BGR ! appsink")
# time_old = time.time()
# if cap.isOpened():
#     cv2.namedWindow("demo", cv2.WINDOW_AUTOSIZE)
#     while True:
#         time_now = time.time()
#         ret_val, img = cap.read()
#         print(1/(time_now - time_old), 'Hz')
#         time_old = time_now
#         cv2.imshow('demo', img)
#         cv2.waitKey(1)
# else:
#     print("Camera open failed")

# cap = cv2.VideoCapture("v4l2src device=/dev/video0 extra-controls=\"c,exposure_auto=3\" ! video/x-raw, width=960, height=540 ! videoconvert ! video/x-raw,format=BGR ! appsink")
# cap = cv2.VideoCapture(1) #works IR
cap = cv2.VideoCapture(2) #works visual range
time_old = time.time()
if cap.isOpened():
    cv2.namedWindow("demo", cv2.WINDOW_AUTOSIZE)
    time_new = time.time()
    ret_val, img = cap.read()
    print("Camera open")
else:
    print("Camera open failed")
cv2.destroyAllWindows()


image = img 
fig, ax = plt.subplots(1)
ax.imshow(image)
plt.show()