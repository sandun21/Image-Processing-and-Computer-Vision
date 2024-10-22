import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_laplace, maximum_filter

f = cv2.imread(r"images\\the_berry_farms_sunflower_field.jpeg", cv2.IMREAD_REDUCED_COLOR_4)   # Read image in grayscale
assert f is not None
gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0   # Normalized grayscale image

sigma = 4.0
log_image = gaussian_laplace(gray, sigma=sigma)

plt.imshow(log_image, cmap='gray')
plt.title('Laplacian of Gaussian')
plt.axis('off')
plt.show()