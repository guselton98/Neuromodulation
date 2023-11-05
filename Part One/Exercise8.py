# Neuromodulation Workshop: EXE 8
# Read from a file and cell plot (including areas)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('cell_properties.csv')

print(data)
print(data.columns)
print(data['X Coor'])

# Store the data into separate variable (you don't have too, this is optional)
x = data['X Coor']
y = data['Y Coor']
area = data['Area']

# Calculate radius area = pi*r^2. Therefore, sqrt(area/pi) = r
# Power in python is **
radius = np.sqrt(area/np.pi)

# Plot circles on an image with radius'
fig, ax = plt.subplots()
for i in range(len(radius)):
    circle = plt.Circle((x[i], y[i]), radius[i], color='b', fill=False)
    ax.add_patch(circle)

ax.set_xlim((0, 10))
ax.set_ylim((0, 10))

plt.show()