# Neuromodulation Workshop Code for Day One
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

# Multiply scalars
A = 1
B = 2.01
C = A*B
print("C is equal to", C)

# Multiply Arrays
A = np.array([1, 5])
B = 2.01
C = A*B
print("C is equal to ", C)

# Write to a file
f = open("temp.txt", "w")
s = "C is equal to " + str(C)
f.write(s)
f.close()

# Read file CellProperties




