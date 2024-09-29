import cv2                  
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("../images/einstein.png",cv2.IMREAD_GRAYSCALE)         # IMREAD Grayscale is essential
assert image is not None, "Image not found"          # Checking if image is loaded

# image = np.array([[1,2,3],[1,2,3],[1,2,3]]).astype(np.uint8) 
 
sobelh = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])


#Using existing filter2D function to apply Sobel filter
filtered_func = cv2.filter2D(image, -1, sobelh)      
  

#Using own code to sobel filter the image
pad_image = np.pad(image, (1), mode='minimum')
filtered_own = np.zeros(image.shape).astype(np.float32)

for i in range(1, image.shape[0]-1):
    for j in range(1, image.shape[1]-1):
        filtered_own[i, j] = np.sum(pad_image[i-1:i+2, j-1:j+2] * sobelh)

def normalize(image):
    max_val = np.max(image)
    abs_image = np.abs(image)
    if max_val != 0:
        norm_image = (abs_image / max_val) * 255
    return norm_image.astype(np.uint8)

filtered_own = np.abs(filtered_own)
max_val = np.max(filtered_own)
if max_val != 0:
    filtered_own = (filtered_own / max_val) * 255
filtered_own = filtered_own.astype(np.uint8)


sobelh1 = np.array([[1], [2], [1]])
sobelh2 = np.array([[1, 0, -1]])

def convImag(image, filter):
    pad_image = np.pad(image, (1), mode='minimum')
    filtered_own = np.zeros(image.shape).astype(np.float32)

    for i in range(1, image.shape[0]-1):
        for j in range(1, image.shape[1]-1):
            filtered_own[i, j] = np.sum(pad_image[i-1:i+2, j-1:j+2] * filter)

    return filtered_own

filtered_own1 = convImag(image, sobelh1)
filtered_own2 = convImag(image, sobelh2)

cv2.imshow("filtered_own1", normalize(filtered_own2))

cv2.imshow("Sobel Filtered Image", filtered_func)  
cv2.imshow("Sobel Filtered Image Own", filtered_own)
cv2.waitKey(0)

                                