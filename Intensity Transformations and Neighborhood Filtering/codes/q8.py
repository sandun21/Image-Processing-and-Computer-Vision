import cv2                  
import numpy as np
import matplotlib.pyplot as plt

#Import All Images
image1 = cv2.imread("../images/a1q5images/im01.png") 
im1_small = cv2.imread("../images/a1q5images/im01small.png")      
image2 = cv2.imread("../images/a1q5images/im02.png") 
im2_small = cv2.imread("../images/a1q5images/im02small.png")
image3 = cv2.imread("../images/a1q5images/im03.png")
im3_small = cv2.imread("../images/a1q5images/im03small.png")
image4 = cv2.imread("../images/a1q5images/taylor.jpg")
im4_small = cv2.imread("../images/a1q5images/taylor_small.jpg")
im4_vsmall = cv2.imread("../images/a1q5images/taylor_very_small.jpg")


nnZoom = lambda img, scale: cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
bilZoom = lambda img, scale: cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

calcSSD = lambda img1, img2: np.sum((img1 - img2)**2) / img1.size

#output ssd values for all images
print("SSD values for all images")
print("SSD between image1 and im1_small under Nearest Neighbour zoom: ", calcSSD(image1, nnZoom(im1_small,4)))
print("SSD between image1 and im1_small under Bilinear zoom: ", calcSSD(image1, bilZoom(im1_small,4)))

print("SSD between image2 and im2_small under Nearest Neighbour zoom: ", calcSSD(image2, nnZoom(im2_small,4)))
print("SSD between image2 and im2_small under Bilinear zoom: ", calcSSD(image2, bilZoom(im2_small,4)))

# print("SSD between image3 and im3_small under Nearest Neighbour zoom: ", calcSSD(image3, nnZoom(im3_small,4)))
# print("SSD between image3 and im3_small under Bilinear zoom: ", calcSSD(image3, bilZoom(im3_small,4)))

print("SSD between image4 and im4_small under Nearest Neighbour zoom: ", calcSSD(image4, nnZoom(im4_small,5)))
print("SSD between image4 and im4_small under Bilinear zoom: ", calcSSD(image4, bilZoom(im4_small,5)))
print("SSD between taylor_small and taylor_very_small under Nearest Neighbour zoom: ", calcSSD(im4_small, nnZoom(im4_vsmall,4)))
print("SSD between taylor_small and taylor_very_small under Bilinear zoom: ", calcSSD(im4_small, bilZoom(im4_vsmall,4)))


