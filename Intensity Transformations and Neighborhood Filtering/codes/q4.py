import cv2                  
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("../images/spider.png")         
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)    
h, s, v = cv2.split(hsv)                          # Split image in to hue, saturation and value

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

a, sigma = 0.75, 70
x = np.arange(0, 256)
t = x + a * 128 * np.exp(-((x - 128) ** 2) / (2 * sigma ** 2))
t = np.clip(t, 0, 255)  
Snew = cv2.LUT(s, t).astype(np.uint8)
t_img = cv2.merge([h, Snew, v])
new_img = cv2.cvtColor(t_img, cv2.COLOR_HSV2BGR)

#plot historgram of original and old image
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.hist(s.ravel(), 256, [0, 256])
plt.title('Original Image')
plt.xlabel('Intensity')
plt.ylabel('Frequency')
plt.subplot(122)
plt.hist(Snew.ravel(), 256, [0, 256])
plt.title('Corrected Image')
plt.xlabel('Intensity')
plt.ylabel('Frequency')
plt.show()

#plot intensity transformation
plt.plot(x, t)
plt.title('Intensity Transformation')
plt.xlabel('Input Intensity')
plt.ylabel('Output Intensity')
plt.show()

cv2.imshow("Original Image", image)
cv2.imshow("Corrected Image", new_img)
#save both images
cv2.imwrite("outputs/q4orig.jpg", image)
cv2.imwrite("outputs/q4corrected.jpg", new_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
