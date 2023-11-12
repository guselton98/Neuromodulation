# NEUROMODULATION WORKSHOP
# PART 2 - EXERCISE 2: USE IMAGE PROCESSING TO REDUCE NOISE

#----------------------------------------------------------------------------------------------------------------------
# IMPORTING LIBRARIES
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
matplotlib.use('TkAgg')
#----------------------------------------------------------------------------------------------------------------------
# READ IN AN IMAGE FILE
filename = 'NLE_s1_contra_GFAP-FITC_NeuN-CY5_20x_1.jpg'
image = plt.imread(filename)

#----------------------------------------------------------------------------------------------------------------------
# SPLIT THE RGB CHANNELS

red_image = image[:, :, 0]      #R
green_image = image[:, :, 1]    #G
blue_image = image[:, :, 2]     #B

#----------------------------------------------------------------------------------------------------------------------
# SHARPEN THE RED CHANNEL

kernel_sharpen_1 = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
sharpened_image = cv2.filter2D(red_image, -1, kernel_sharpen_1)

#----------------------------------------------------------------------------------------------------------------------
# CREATING AND DISPLAYING THE UNSHARPENED AND SHARPENED FIGURES

plt.figure(figsize=(12, 3))

plt.subplot(121)
plt.imshow(red_image)
plt.title("Unsharpened Red Channel")

plt.subplot(122)
plt.imshow(sharpened_image)
plt.title("Sharpened Red Channel")

#----------------------------------------------------------------------------------------------------------------------
# SHOW ALL OF THE FIGURES

plt.show()