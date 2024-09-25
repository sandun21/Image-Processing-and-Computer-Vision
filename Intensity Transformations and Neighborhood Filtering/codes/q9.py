import cv2                  
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("../images/daisy.jpg")         # As applying gamma correction on L plane
L, a, b = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2LAB))     