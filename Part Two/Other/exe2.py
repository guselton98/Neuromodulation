# Neuromodulation Workshop: EXE 2
import matplotlib.pyplot as plt
import numpy as np

# Read in image file
filename = '../NLE_s1_contra_GFAP-FITC_NeuN-CY5_20x_1.jpg'
image = plt.imread(filename)
M = np.asarray(image)

# Plot separate RGB channels of image
red_image = M[:, :, 0]      #R
green_image = M[:, :, 1]    #G
blue_image = M[:, :, 2]     #B

plt.figure(figsize=(12, 6))
plt.subplot(131)
plt.imshow(red_image, cmap='Reds', vmin=0, vmax=255)
plt.title("Red Channel")
plt.subplot(132)
plt.imshow(green_image, cmap='Greens', vmin=0, vmax=255)
plt.title("Green Channel")
plt.subplot(133)
plt.imshow(blue_image, cmap='Blues', vmin=0, vmax=255)
plt.title("Blue Channel")
plt.show()
