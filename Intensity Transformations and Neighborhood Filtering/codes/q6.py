import cv2                  
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("../images/jeniffer.jpg")        # As applying gamma correction on L plane
assert image is not None
h, s, v = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL))

# Show 3 channels in grayscale in matplotlib plots
plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.imshow(h, cmap='gray')
plt.title('Hue')
plt.subplot(132)
plt.imshow(s, cmap='gray')
plt.title('Saturation')
plt.subplot(133)
plt.imshow(v, cmap='gray')
plt.title('Value')
plt.show()


#Use saturation plane to extract foreground mask by thresholding
_, mask = cv2.threshold(s, 12, 255, cv2.THRESH_BINARY)
plt.imshow(mask, cmap='gray')
plt.title('Foreground mask')
plt.show()

#Obtain foreground using cv2.bitwise
foreground = cv2.bitwise_and(v, mask)
plt.imshow(foreground, cmap='gray')
plt.title('Foreground')
plt.show()

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


newim = histEqualization(foreground)
plt.imshow(newim, cmap='gray')
plt.title('Foreground')
plt.show()


_, mask = cv2.threshold(s, 0, 12, cv2.THRESH_BINARY)
mask = cv2.bitwise_and(v, mask)
plt.imshow(mask, cmap='gray')
plt.title('Foreground mask')
plt.show()