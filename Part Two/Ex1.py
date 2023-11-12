# NEUROMODULATION WORKSHOP
# PART 2 - EXERCISE 1: DISPLAY AND SEPARATE AN RGB IMAGE

#----------------------------------------------------------------------------------------------------------------------
# IMPORTING LIBRARIES
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

#----------------------------------------------------------------------------------------------------------------------
# READ IN AN IMAGE FILE
filename = 'NLE_s1_contra_GFAP-FITC_NeuN-CY5_20x_1.jpg'
image = plt.imread(filename)

#----------------------------------------------------------------------------------------------------------------------
# CREATING AND DISPLAYING THE FIGURE
plt.imshow(image)
plt.title(filename)

#----------------------------------------------------------------------------------------------------------------------
# SPLIT THE RGB CHANNELS

red_image = image[:, :, 0]      #R
green_image = image[:, :, 1]    #G
blue_image = image[:, :, 2]     #B

#----------------------------------------------------------------------------------------------------------------------
# CREATING AND DISPLAYING THE FIGURE WITH COLOUR MAPPING

plt.figure(figsize=(12, 3))

plt.subplot(131)
plt.imshow(red_image, cmap='Reds')
plt.title("Red Channel")

plt.subplot(132)
plt.imshow(green_image, cmap='Greens')
plt.title("Green Channel")

plt.subplot(133)
plt.imshow(blue_image, cmap='Blues')
plt.title("Blue Channel")

#----------------------------------------------------------------------------------------------------------------------
# SHOW ALL OF THE FIGURES

plt.show()