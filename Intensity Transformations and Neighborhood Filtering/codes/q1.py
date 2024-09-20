import cv2
import numpy as np
import matplotlib.pyplot as plt

f = cv2.imread("../images/emma.jpg", cv2.IMREAD_GRAYSCALE)   # Read image in grayscale

arr1 = np.linspace(0,50,50,dtype='uint8')
arr2 = np.linspace(100,256,100,dtype='uint8')
arr3 = np.linspace(150,256,106,dtype='uint8')

# Lookup table for the transformation
t = np.concatenate((arr1, arr2, arr3),dtype='uint8')        # Concatenate arrays

g = cv2.LUT(f, t)                                           # Apply intensity transformation

cv2.imshow("Original Image", f)
cv2.imshow("Transformed Image", g)
cv2.waitKey(0)

    

