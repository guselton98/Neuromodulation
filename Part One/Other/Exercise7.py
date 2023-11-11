# Neuromodulation Workshop: EXE 7
# Read from a file and 2D density plot

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('../cell_properties.csv')

print(data)
print(data.columns)
print(data['X Coor'])

# Store the data into separate variable (you don't have too, this is optional)
x = data['X Coor']
y = data['Y Coor']
area = data['Area']

# Create histograms plots for each variable
xbin = np.linspace(0, 10, num=100)
ybin = np.linspace(0, 10, num=100)
areabin = np.linspace(0, 0.3, num=100)

plt.figure()
plt.subplot(131)
plt.hist(x, xbin)
plt.title('X Coor')
plt.subplot(132)
plt.hist(y, ybin)
plt.title('Y Coor')
plt.subplot(133)
plt.hist(area, areabin)
plt.title('Area')

plt.show()

# 2D Histogram plot :
#   https://numpy.org/doc/stable/reference/generated/numpy.histogram2d.html#numpy-histogram2d
xedges = np.linspace(0, 10, num=100)
yedges = np.linspace(0, 10, num=100)
H, xedges, yedges = np.histogram2d(x, y, bins=(xedges, yedges))
H = H.T
fig = plt.figure(figsize=(7, 3))
ax = fig.add_subplot(131, title='Density Plot of Cells')
plt.imshow(H, interpolation='nearest', origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
plt.show()