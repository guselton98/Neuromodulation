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

image = plt.imread('NLE_s1_contra_GFAP-FITC_NeuN-CY5_20x_1.jpg')
M = np.asarray(image)
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

#plt.show()

# Sharpen red channel
kernel_sharpen_1 = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
sharpened_image = cv2.filter2D(M[:, :, 0], -1, kernel_sharpen_1)
plt.imshow(sharpened_image)
plt.title("Sharpened Image")
#plt.show()

# Apply a threshold to the sharpened image to create a mask
ret, thres_image = cv2.threshold(sharpened_image, 50, 255,cv2.THRESH_BINARY)
plt.imshow(thres_image)
plt.title("Masked Image")
#plt.show()
#cv2.waitKey(0)

# Apply open/close morphologies
kernel = np.ones((3, 3), np.uint8)
closed_image = cv2.morphologyEx(thres_image, cv2.MORPH_CLOSE,
                           kernel, iterations=1)

kernel = np.ones((5,5), np.uint8)
opened_image = cv2.morphologyEx(closed_image, cv2.MORPH_OPEN,
                           kernel, iterations=1)

plt.imshow(opened_image)
plt.title("Opening")
#plt.show()

Red_channel = M[:, :, 0]
Mask = opened_image
result_image = cv2.bitwise_and(Red_channel, Mask)
plt.imshow(result_image)
plt.title("result_image")
plt.show()

label_img = label(opened_image)
regions = regionprops(label_img)

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

props = regionprops_table(label_img, properties=('centroid',
                                                 'orientation',
                                                 'solidity'))

df = pd.DataFrame(props)

df.to_csv('temp.csv')