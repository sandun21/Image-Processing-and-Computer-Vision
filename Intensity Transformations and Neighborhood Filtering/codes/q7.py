import cv2                  
import time
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("../images/einstein.png",cv2.IMREAD_GRAYSCALE)         # IMREAD Grayscale is essential
assert image is not None, "Image not found"          # Checking if image is loaded

sobelh = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])

start1 = time.time()
#Using existing filter2D function to apply Sobel filter
filtered_func = cv2.filter2D(image, -1, sobelh)      
  


#Using own code to sobel filter the image
def convImag(image, filter):
    pad_image = np.pad(image, (1), mode='minimum')
    filtered_own = np.zeros(image.shape).astype(np.float32)

    for i in range(1, image.shape[0]-1):
        for j in range(1, image.shape[1]-1):
            filtered_own[i, j] = np.sum(pad_image[i-1:i+2, j-1:j+2] * filter)

    return filtered_own

def normalize(image):
    '''
    Normalizes the image to 0-255
    '''
    max_val = np.max(image)
    abs_image = np.abs(image)
    if max_val != 0:
        norm_image = (abs_image / max_val) * 255
    return norm_image.astype(np.uint8)

end1 = time.time()
filtered_own = normalize(convImag(image, sobelh))
end2 = time.time()



#Using seperable property of Sobel filter
sobelh1 = np.array([[1], [2], [1]])
sobelh2 = np.array([[1, 0, -1]])

pad_image = np.pad(image, (1), mode='minimum')
filtered_own1 = np.zeros(image.shape).astype(np.float32)


#convolve with sobelh1 column array
for i in range(1, image.shape[0]-1):
    filtered_own1[i] = np.sum(pad_image[i-1:i+2,1:-1] * sobelh1, axis=0)
    
for j in range(1, image.shape[1]-1):
    filtered_own1[:,j] = np.sum(pad_image[1:-1,j-1:j+2] * sobelh2, axis=1)
filtered_own1 = normalize(filtered_own1)
end3 = time.time()


#Displaying images
cv2.imshow("Sobel Filtered Image", filtered_func)  
cv2.imshow("Sobel Filtered Image Own", filtered_own*2)
cv2.imshow("Sobel Filtered Image Own with using seperability property", filtered_own1*9)
cv2.waitKey(0)

#Printing time taken
print("Time taken by filter2D function: ", end1-start1)
print("Time taken by own code: ", end2-end1)
print("Time taken by seperable property: ", end3-end2)
                                