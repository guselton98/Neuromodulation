import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import scipy.ndimage # useful resource aswell
from skimage.measure import label, regionprops, regionprops_table

from skimage.draw import ellipse
import skimage
from skimage.measure import label, regionprops, regionprops_table
from skimage.transform import rotate

# Tiff import/export
from PIL import Image

# Read in image file
image = plt.imread('Part Two\\NLE_s1_contra_GFAP-FITC_NeuN-CY5_20x_1.jpg')
M = np.asarray(image)

# Plot separate RGB channels of image
plt.figure(figsize=(12, 6))
plt.subplot(131)
plt.imshow(M[:, :, 0], cmap='Reds', vmin=0, vmax=255)
plt.title("Red Channel")
plt.subplot(132)
plt.imshow(M[:, :, 1], cmap='Greens', vmin=0, vmax=255)
plt.title("Green Channel")
plt.subplot(133)
plt.imshow(M[:, :, 2], cmap='Blues', vmin=0, vmax=255)
plt.title("Blue Channel")

# Sharpen red channel
plt.figure(figsize=(12, 6))
kernel_sharpen_1 = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
sharpened_image = cv2.filter2D(M[:, :, 0], -1, kernel_sharpen_1)
plt.imshow(sharpened_image)
plt.title("Sharpened Image")

# Apply a threshold to the sharpened image to create a mask
ret, thres_image = cv2.threshold(sharpened_image, 60, 255,cv2.THRESH_BINARY)

def diamond(r):
    b = np.r_[:r, r:-1:-1]
    return (b[:, None]+b) >= r

# Delete small specs
spec_mask = diamond(3)*1
spec_mask = spec_mask.astype(np.uint8)
thres_image = cv2.morphologyEx(thres_image, cv2.MORPH_OPEN,
                           spec_mask, iterations=1)
thres_image = cv2.morphologyEx(thres_image, cv2.MORPH_CLOSE,
                           spec_mask, iterations=1)

# Close Image
close_mask = diamond(5)*1
close_mask = close_mask.astype(np.uint8)
closed_image = cv2.morphologyEx(thres_image, cv2.MORPH_CLOSE,
                           close_mask, iterations=1)
# Open Image
open_mask = diamond(3)*1
open_mask = open_mask.astype(np.uint8)
opened_image = cv2.morphologyEx(closed_image, cv2.MORPH_OPEN,
                           open_mask, iterations=1)

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

# Overlay opened_image with red channel
Red_channel = M[:, :, 0]
result_image = cv2.bitwise_and(Red_channel, opened_image)

plt.figure()
plt.subplot(121)
plt.imshow(result_image, cmap='Reds', vmin=0, vmax=255)
plt.title('Final Image')
plt.subplot(122)
plt.imshow(image)
plt.title("Original Image")
plt.show()

# Label Image and find region properties
label_img = label(opened_image)
regions = regionprops(label_img)

# Visualise results from regionprops
plt.figure()
fig, ax = plt.subplots()
ax.imshow(opened_image, cmap=plt.cm.gray)
for props in regions:
    y0, x0 = props.centroid
    orientation = props.orientation
    x1 = x0 + math.cos(orientation) * 0.5 * props.minor_axis_length
    y1 = y0 - math.sin(orientation) * 0.5 * props.minor_axis_length
    x2 = x0 - math.sin(orientation) * 0.5 * props.major_axis_length
    y2 = y0 - math.cos(orientation) * 0.5 * props.major_axis_length

    ax.plot((x0, x1), (y0, y1), '-r', linewidth=2.5)
    ax.plot((x0, x2), (y0, y2), '-r', linewidth=2.5)
    ax.plot(x0, y0, '.g', markersize=15)
plt.show()

# Overlay with Red image

# Overlay with original Image


# Export regionprops data to excel spreadsheet
props = regionprops_table(label_img, properties=('centroid',
                                                 'orientation',
                                                 'solidity'))
df = pd.DataFrame(props)
df.to_csv('Part Two\\temp.csv')