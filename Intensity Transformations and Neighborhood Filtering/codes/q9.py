import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load the image
image = cv2.imread('../images/daisy.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for matplotlib display

# Step 2: Initialize the mask, background and foreground models
mask = np.zeros(image.shape[:2], np.uint8)  # Create a mask initialized to 0
bgd_model = np.zeros((1, 65), np.float64)  # Background model (needed for GrabCut)
fgd_model = np.zeros((1, 65), np.float64)  # Foreground model (needed for GrabCut)

# Step 3: Define the rectangle around the foreground (flower)
# You might need to adjust the rectangle coordinates depending on the flower position in the image
rect = (50, 50, image.shape[1]-10, image.shape[0]-300)

# Step 4: Apply GrabCut
cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

# Step 5: Modify the mask for foreground and background
# Assign 1 for foreground, and 0 for background
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

# Extract the foreground and background
foreground = image_rgb * mask2[:, :, np.newaxis]
background = image_rgb * (1 - mask2[:, :, np.newaxis])

# Step 6: Show the segmentation mask, foreground, and background
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title("Segmentation Mask")
plt.imshow(mask2, cmap='gray')

plt.subplot(1, 3, 2)
plt.title("Foreground Image")
plt.imshow(foreground)

plt.subplot(1, 3, 3)
plt.title("Background Image")
plt.imshow(background)

plt.show()

# (b) Producing the enhanced image with a blurred background

# Step 7: Blur the background
blurred_background = cv2.GaussianBlur(image_rgb, (21, 21), 0)

# Step 8: Combine blurred background with the foreground
enhanced_image = blurred_background.copy()
enhanced_image[mask2 == 1] = foreground[mask2 == 1]  # Keep the foreground unchanged

# Step 9: Show original and enhanced images side by side
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image_rgb)

plt.subplot(1, 2, 2)
plt.title("Enhanced Image with Blurred Background")
plt.imshow(enhanced_image)

plt.show()

# (c) Explanation:
# The background near the edges of the flower appears dark because the segmentation might not be perfect.
# Pixels close to the boundary of the flower could be classified as background, which results in an abrupt change 
# from foreground to background, leading to a dark halo effect or misclassification along the edges.
