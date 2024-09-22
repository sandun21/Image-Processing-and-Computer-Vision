import cv2                  
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("../images/highlights_and_shadows.jpg")         # As applying gamma correction on L plane
L, a, b = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2LAB))        # Split the L*, a*, b* channels

gamma = 5
t = (np.arange(0,256)/255)**gamma * 255
L_corrected = cv2.LUT(L, t).astype(np.uint8)


corrected_imglab = cv2.merge([L_corrected, a, b])
corrected_img = cv2.cvtColor(corrected_imglab, cv2.COLOR_LAB2BGR)

cv2.imshow("Original Image", image)
cv2.imshow("Corrected Image", corrected_img)

#show historgrams
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.hist(L.ravel(), 256, [0, 256])
plt.title('Original Image')
plt.xlabel('Intensity')
plt.ylabel('Frequency')
plt.subplot(122)
plt.hist(L_corrected.ravel(), 256, [0, 256])
plt.title('Corrected Image')
plt.xlabel('Intensity')
plt.ylabel('Frequency')
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
