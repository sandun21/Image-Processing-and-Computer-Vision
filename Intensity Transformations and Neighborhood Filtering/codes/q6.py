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


#saturation plane is better to extract foreground mask
# obtain foreground using cv2.bitwise_and
