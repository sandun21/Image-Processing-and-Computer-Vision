import cv2                                     # https://medium.com/@susanne.schmid/visualization-of-medical-images-adjusting-contrast-through-windowing-c2dd9abb1d5
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("../images/brain_proton_density_slice.png")   # Read image in grayscale

warr1 = np.linspace(0,20,100,dtype='uint8')
warr2 = np.linspace(20,80,80,dtype='uint8')                # If we use 256 as upper range it goes to 0
warr3 = np.linspace(80, 255, 76,dtype='uint8')

# garr1 = np.linspace(0,100,50,dtype='uint8')
# garr2 = np.linspace(100,150,50,dtype='uint8')                # If we use 256 as upper range it goes to 0
# garr3 = np.linspace(150,255,156,dtype='uint8')

garr1 = np.linspace(0,10,20,dtype='uint8')
garr2 = np.linspace(20,250,180,dtype='uint8')                # If we use 256 as upper range it goes to 0
garr3 = np.linspace(0,20,56,dtype='uint8')

# Lookup table for the transformation
t_wh = np.concatenate((warr1, warr2, warr3),dtype='uint8')        # Concatenate arrays
t_gr = np.concatenate((garr1, garr2, garr3),dtype='uint8')

g = cv2.LUT(image, t_gr)                                           # Apply intensity transformation
w = cv2.LUT(image, t_wh) 


#plot intensity historgram
plt.hist(image.ravel(),256,[0,256],label='Original Image')
plt.legend()

# Display transformation in plots 
fig, ax = plt.subplots(1,5, figsize=(20,5))
ax[0].imshow(image, cmap='gray')
ax[0].set_title("Original Image")
ax[1].plot(t_wh, label='transformation')
ax[1].set_xlabel('Input Intensity')
ax[1].set_ylabel('Output Intensity')
ax[1].set_title("White Transformation")
ax[2].imshow(w, cmap='gray')
ax[2].set_title("Accentuate White Matter")
ax[3].plot(t_gr, label='transformation')
ax[3].set_title("Gray Transformation")
ax[4].imshow(g, cmap='gray')
ax[4].set_title("Accentuate Gray Matter")
plt.show()