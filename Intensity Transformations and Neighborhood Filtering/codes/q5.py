import cv2                  
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("../images/shells.tif")     

#find frequency of occurance of each pixel value in 2D array without using historgram function 
unique, counts = np.unique(image, return_counts=True)
frequency = dict(zip(unique, counts))
numpx = np.prod(image.shape)
multfactor = 255/numpx
hist = np.array([frequency[i] if i in frequency.keys() else 0 for i in range(256)])

print("earlier historgram", hist)
cdf = np.cumsum(hist)
t = np.floor(cdf * multfactor).astype(np.uint8)
newim = t[image]

print("new historgram", t)
cv2.imshow("Original Image", newim)
cv2.waitKey(0)  