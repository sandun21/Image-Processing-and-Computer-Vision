import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_laplace, maximum_filter

f = cv2.imread(r"images\\the_berry_farms_sunflower_field.jpeg", cv2.IMREAD_REDUCED_COLOR_4)   # Read image in grayscale
assert f is not None
gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0   # Normalized grayscale image

sigma = 8
log_image = gaussian_laplace(gray, sigma=sigma)
log_image = (sigma ** 2) * np.abs(log_image)

threshold = 0.30
neighborhood_size = 10

# Detect local maxima
local_max = maximum_filter(log_image, size=neighborhood_size) == log_image
blobs = (log_image > threshold) & local_max

# Get coordinates of blobs
y, x = np.nonzero(blobs)
r = np.sqrt(2) * sigma

# Plot blobs
fig, ax = plt.subplots()
ax.imshow(gray, cmap='gray')
for y_i, x_i in zip(y, x):
    c = plt.Circle((x_i, y_i), r, color='red', linewidth=0.5, fill=False)
    ax.add_patch(c)
ax.set_axis_off()
plt.show()
