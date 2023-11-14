# Neuromodulation Workshop: EXE 6
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.measure import label

from skimage import morphology
matplotlib.use('TkAgg')

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
# Erode Image
close_mask = diamond(3)*1
close_mask = close_mask.astype(np.uint8)
closed_image = cv2.morphologyEx(closed_image, cv2.MORPH_ERODE, close_mask, iterations=1)

# Label Image and
label_img = label(closed_image)
# Remove small specs from labelled image
cleaned_img = morphology.remove_small_objects(label_img, min_size=20*np.pi**2, connectivity=2)

plt.figure()
plt.subplot(121)
plt.imshow(label_img)
plt.title('Labelled Image')
plt.subplot(122)
plt.imshow(cleaned_img)
plt.title("Spec Removal")
plt.show()
