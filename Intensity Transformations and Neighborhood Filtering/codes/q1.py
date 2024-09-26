import cv2
import numpy as np

f = cv2.imread("../images/emma.jpg", cv2.IMREAD_GRAYSCALE)   # Read image in grayscale
assert f is not None

arr1 = np.linspace(0,50,50,dtype='uint8')
arr2 = np.linspace(100,255,100,dtype='uint8')
arr3 = np.linspace(150,255,106,dtype='uint8')

# Lookup table for the transformation
t = np.concatenate((arr1, arr2, arr3),dtype='uint8')        # Concatenate arrays
g = cv2.LUT(f, t)                                           # Apply intensity transformation

#save two images
cv2.imwrite("outputs/q1emma_transformed.jpg", g)
cv2.imwrite("outputs/q1emma.jpg", f)

cv2.imshow("Original Image", f)
cv2.imshow("Transformed Image", g)
cv2.waitKey(0)

