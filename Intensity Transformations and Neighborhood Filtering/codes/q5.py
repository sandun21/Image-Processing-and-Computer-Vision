import cv2                  
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("../images/shells.tif")     
assert image is not None

def historgram(image):
    '''
       function that takes image as the input and returns the frequency of occurance of each pixel value
    '''
    unique, counts = np.unique(image, return_counts=True)
    frequency = dict(zip(unique, counts))
    hist = np.array([frequency[i] if i in frequency.keys() else 0 for i in range(256)])
    return hist

def histEqualization(image):
    '''
       function that takes image to be equalized as the input and returns the equalized image 
    '''
    numpx = np.prod(image.shape)    
    multfactor = 255/numpx
    cdf = np.cumsum(historgram(image))
    
    t = np.floor(cdf * multfactor).astype(np.uint8)                
    newimg = t[image]
    return newimg


newim = histEqualization(image)

#historgrams before and after normalization
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.hist(image.ravel(), 256, [0, 256])
plt.title('Original Image')
plt.xlabel('Intensity')
plt.ylabel('Frequency')
plt.subplot(122)
plt.hist(newim.ravel(), 256, [0, 256])
plt.title('Equalized Image')
plt.xlabel('Intensity')
plt.ylabel('Frequency')
plt.show()

cv2.imshow("Equilized Image", newim)
cv2.imwrite("outputs/q5orig.jpg", image)
cv2.imwrite("outputs/q5corrected.jpg", newim)
cv2.waitKey(0)  