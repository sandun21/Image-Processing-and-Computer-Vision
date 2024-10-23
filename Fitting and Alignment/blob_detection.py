import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_laplace, maximum_filter

# Read the image
f = cv2.imread(r"images\\the_berry_farms_sunflower_field.jpeg", cv2.IMREAD_REDUCED_COLOR_4)  # Read image in reduced color
assert f is not None, "Image not found."

# Convert to grayscale and normalize
gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0


sigma_values = range(180,210,3) # sigma values
colors = colors = ['red','green','blue','yellow','purple','orange','cyan','magenta','lime','pink']  # Assign a unique color for each sigma

# Parameters
threshold = 0.1
neighborhood_size = 10


def LoG(s):
    hw = round(3*s)                  # For gaussian to be spreaded fully over kernel
    X, Y = np.meshgrid(np.arange(-hw, hw + 1, 1), np.arange(-hw, hw + 1, 1))
    log = ((X**2 + Y**2)/(2*s**2) - 1) * np.exp(-(X**2 + Y**2)/(2*s**2)) / (np.pi * s**4)
    return log * s**2



# Prepare the plot
fig, ax = plt.subplots()
rgb_image = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
ax.imshow(rgb_image)

# Iterate over each sigma and plot corresponding circles
for sigma, color in zip(sigma_values, colors):
    log_image = cv2.filter2D(gray, -1, LoG(s=sigma))

    # Detect local maxima
    local_max = maximum_filter(log_image, size=neighborhood_size) == log_image
    blobs = (log_image > threshold) & local_max

    # Get coordinates of blobs
    y, x = np.nonzero(blobs)
    r = np.sqrt(2) * sigma

    # Plot circles for current sigma
    for y_i, x_i in zip(y, x):
        circle = plt.Circle((x_i, y_i), r, color=color, linewidth=2, fill=False, alpha=1)
        ax.add_patch(circle)

# Optional: Create a legend
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], color=colors[i], lw=2, label=f'Sigma {sigma_values[i]}') for i in range(len(sigma_values))]
ax.legend(handles=legend_elements, loc='upper right')

# Finalize the plot
ax.set_axis_off()
plt.show()
