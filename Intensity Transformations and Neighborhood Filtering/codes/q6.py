import cv2                  
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("../images/jeniffer.jpg")        # As applying gamma correction on L plane
assert image is not None
h, s, v = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL))

_, mask = cv2.threshold(s, 12, 255, cv2.THRESH_BINARY)             #Use saturation plane to extract foreground mask
foreground = cv2.bitwise_and(v, mask)                              #Obtain foreground using cv2.bitwise
hist = cv2.calcHist([v], [0], mask, [256], [0, 256])               #Calculate the histogram of the Value channel

cdf = hist.cumsum()

# Plot the histogram cumalative sum
plt.figure()
plt.plot(cdf)
plt.title('Cumulative Sum of the Histogram')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.show()

pixels = cdf[-1]
#Transfomation for histogram equalization
t = np.array([(256-1)/(pixels)*cdf[k] for k in range(256)]).astype("uint8")
vn = t[v]

# Calculate the histogram of the equalized Value channel
hist = cv2.calcHist([v], [0], mask, [256], [0, 256])

# Plot the histogram befor and after equalization in two plots
plt.subplot(121)
plt.bar(np.arange(len(hist)), hist.flatten())
plt.title('Histogram of Value for Foreground')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.subplot(122)
plt.bar(np.arange(len(hist)), cv2.calcHist([vn], [0], mask, [256], [0, 256]).flatten())
plt.title('Histogram of Equalized Value for Foreground')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.show()


# Merge
merged = cv2.merge([h, s, vn])
foreground_modified = cv2.cvtColor(merged, cv2.COLOR_HSV2RGB)

background = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(mask))
result = cv2.add(cv2.cvtColor(background, cv2.COLOR_BGR2RGB), foreground_modified)

#Show the hue, saturation, and value plane, the mask, the original image, and the result with the histogramequalized foreground
plt.figure(figsize=(15, 15))
plt.subplot(231)
plt.imshow(h, cmap='gray')
plt.title('Hue')
plt.subplot(232)
plt.imshow(s, cmap='gray')
plt.title('Saturation')
plt.subplot(233)
plt.imshow(v, cmap='gray')
plt.title('Value')
plt.subplot(234)
plt.imshow(mask, cmap='gray')
plt.title('Foreground Mask')
plt.subplot(235)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.subplot(236)
plt.imshow(result)
plt.title('Result')
plt.show()






# Show 3 channels in grayscale in matplotlib plots
# plt.figure(figsize=(15, 5))
# plt.subplot(131)
# plt.imshow(h, cmap='gray')
# plt.title('Hue')
# plt.subplot(132)
# plt.imshow(s, cmap='gray')
# plt.title('Saturation')
# plt.subplot(133)
# plt.imshow(v, cmap='gray')
# plt.title('Value')
# plt.show()