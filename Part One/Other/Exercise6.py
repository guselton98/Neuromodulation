# Neuromodulation Workshop: EXE 6
# Read from a file and histogram

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
plt.figure()
plt.subplot(131)
plt.hist(x)
plt.title('X Coor')
plt.subplot(132)
plt.hist(y)
plt.title('Y Coor')
plt.subplot(133)
plt.hist(area)
plt.title('Area')

# A bit hard to see... decrease bin size of histogram plot
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
