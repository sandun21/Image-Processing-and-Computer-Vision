import cv2                  
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("../images/spider.png")         
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)    
h, s, v = cv2.split(hsv)                          # Split image in to hue, saturation and value

a = 0.9
sigma = 70
x = np.arange(0, 256)
t = x + a * 128 * np.exp(-((x - 128) ** 2) / (2 * sigma ** 2))
t = np.clip(t, 0, 255)  
Snew = cv2.LUT(s, t).astype(np.uint8)
new_img = cv2.merge([h, Snew, v])
new_img = cv2.cvtColor(new_img, cv2.COLOR_HSV2BGR)


cv2.imshow("Original Image", image)
cv2.imshow("Corrected Image", new_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
