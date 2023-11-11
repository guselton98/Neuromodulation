# Neuromodulation Workshop: EXE 1
import matplotlib.pyplot as plt
import numpy as np

# Read in image file
filename = '../NLE_s1_contra_GFAP-FITC_NeuN-CY5_20x_1.jpg'
image = plt.imread(filename)

# Convert to numpy array as it is easier to work with
M = np.asarray(image)

# Show the image
plt.imshow(image)
plt.title(filename)
plt.show()