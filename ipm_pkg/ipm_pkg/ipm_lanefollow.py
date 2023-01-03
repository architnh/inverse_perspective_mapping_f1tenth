# -*- coding: utf-8 -*-
import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('lane.png', cv2.COLOR_BGR2RGB)
plt.imshow(img)

blur_image = cv2.GaussianBlur(img, (5,5), 2)
plt.imshow(blur_image)

hsv = cv2.cvtColor(blur_image, cv2.COLOR_BGR2HSV)
plt.imshow(hsv)

mask = cv2.inRange(hsv, np.array([20, 70, 70]), np.array([40, 255, 255]))
# Bitwise-AND mask and original image
res = cv2.bitwise_and(hsv,hsv, mask= mask)

#plt.imshow(mask)
plt.imshow(res)

contours, hierarchy = cv2.findContours(image=mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

# draw contours on the original image
image_copy = img.copy()
cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=5, lineType=cv2.LINE_AA)
# see the results
plt.imshow(image_copy)
cv2.waitKey(0)
cv2.imwrite('lane_detection.jpg', image_copy)
cv2.destroyAllWindows()