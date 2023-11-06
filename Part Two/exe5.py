# Neuromodulation Workshop: EXE 5
import matplotlib.pyplot as plt
import numpy as np
import cv2


# Read in image file
filename = 'NLE_s1_contra_GFAP-FITC_NeuN-CY5_20x_1.jpg'
image = plt.imread(filename)
M = np.asarray(image)

red_image = M[:, :, 0]      #R
green_image = M[:, :, 1]    #G
blue_image = M[:, :, 2]     #B

# Sharpen red channel
kernel_sharpen_1 = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
sharpened_image = cv2.filter2D(red_image, -1, kernel_sharpen_1)

# Apply a threshold to the sharpened image to create a mask
ret, thres_image = cv2.threshold(sharpened_image, 60, 255,cv2.THRESH_OTSU)

# Diamond function: https://stackoverflow.com/questions/58348401/numpy-array-filled-in-diamond-shape
def diamond(r):
    b = np.r_[:r, r:-1:-1]
    return (b[:, None]+b) >= r

# Open Image
open_mask = diamond(3)*1
open_mask = open_mask.astype(np.uint8)
opened_image = cv2.morphologyEx(thres_image, cv2.MORPH_OPEN, open_mask, iterations=1)
# Close Image
close_mask = diamond(5)*1
close_mask = close_mask.astype(np.uint8)
closed_image = cv2.morphologyEx(opened_image, cv2.MORPH_CLOSE, close_mask, iterations=1)

# Plot the results of opening and closing a thresholded image
plt.figure()
plt.subplot(131)
plt.imshow(thres_image)
plt.title("Thresholded Image")
plt.subplot(132)
plt.imshow(closed_image)
plt.title("Closed Image")
plt.subplot(133)
plt.imshow(opened_image)
plt.title("Opened Image")
plt.show()

# Overlay opened_image with red channel
result_image = cv2.bitwise_and(red_image, closed_image)

plt.figure()
plt.subplot(121)
plt.imshow(result_image, cmap='Reds', vmin=0, vmax=255)
plt.title('Final Image')
plt.subplot(122)
plt.imshow(image)
plt.title("Original Image")
plt.show()

