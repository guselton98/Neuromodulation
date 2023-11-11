# Neuromodulation Workshop: EXE 3
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Read in image file
filename = '../NLE_s1_contra_GFAP-FITC_NeuN-CY5_20x_1.jpg'
image = plt.imread(filename)
M = np.asarray(image)

red_image = M[:, :, 0]      #R
green_image = M[:, :, 1]    #G
blue_image = M[:, :, 2]     #B

# Sharpen red channel
kernel_sharpen_1 = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
sharpened_image = cv2.filter2D(red_image, -1, kernel_sharpen_1)

# Show the sharpened image
plt.figure(figsize=(12, 6))
plt.imshow(sharpened_image)
plt.title("Sharpened Image")

# What happens to the boundaries of the cells ?