# NEUROMODULATION WORKSHOP
# PART 3 - EXERCISE 3: DIFFERENT STYLES OF PLOTTING

#----------------------------------------------------------------------------------------------------------------------
# IMPORTING LIBRARIES
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.colors as clr

#----------------------------------------------------------------------------------------------------------------------
# READ IN FILE
data = pd.read_csv('cell_properties.csv')

#----------------------------------------------------------------------------------------------------------------------
# STORE INFORMATION
print(data.columns)

# Store the data into separate variables
x = data['X Coor']
y = data['Y Coor']
area = data['Area']

#----------------------------------------------------------------------------------------------------------------------
# HISTOGRAM PLOT

# Increasing discreet sampling to 100
xbin = np.linspace(0, 10, num=100)
ybin = np.linspace(0, 10, num=100)
areabin = np.linspace(0, 0.3, num=100)

plt.figure()
plt.suptitle('Frequency Analysis')
plt.subplot(131)
plt.hist(x, xbin)
plt.title('X Coor')
plt.subplot(132)
plt.hist(y, ybin)
plt.title('Y Coor')
plt.subplot(133)
plt.hist(area, areabin)
plt.title('Area')

#----------------------------------------------------------------------------------------------------------------------
# SCATTER PLOT

# Plot a map of the cells using a scatter plot
plt.figure()
plt.scatter(x, y, marker="o")
plt.title("Plot of cell locations recorded")
plt.legend(["Cell Location"])
plt.xlabel("Longitude")
plt.xlabel("Latitude")

#----------------------------------------------------------------------------------------------------------------------
# SCATTER PLOT WITH CIRCLES OF A DEFINED DIAMETER

radius = np.sqrt(area/np.pi)   # Setting the diameter for the circles

fig, ax = plt.subplots()   # Creating the figure and parameters
for i in range(len(radius)):
    circle = plt.Circle((x[i], y[i]), radius[i], color='b', fill=False)
    ax.add_patch(circle)

ax.set_xlim((0, 10))
ax.set_ylim((0, 10))

#----------------------------------------------------------------------------------------------------------------------
# DENSITY PLOT: 2D HISTOGRAM PLOT
# This one involves some mathematical concepts and slightly more advanced Python understanding
# https://numpy.org/doc/stable/reference/generated/numpy.histogram2d.html#numpy-histogram2d

xedges = np.linspace(0, 10, num=100)
yedges = np.linspace(0, 10, num=100)
H, xedges, yedges = np.histogram2d(x, y, bins=(xedges, yedges))   # Something exclusive to Python are 'tuples'

H = H.T   # 'T' transposes the matrix (2D array) 'H'

#----------------------------------------------------------------------------------------------------------------------
# CREATING AND DISPLAYING THE FIGURE WITH CERTAIN PARAMETERS

fig, ax = plt.subplots()   # Creating the figure and parameters
h = ax.hist2d(x, y, bins=100, norm=clr.LogNorm())   # Capturing the colour bar from the data
fig.colorbar(h[-1], ax=ax)   # Applying the colour bar to the figure
plt.title("Density Plot of Cells")
plt.imshow(H, interpolation='nearest', origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])

#----------------------------------------------------------------------------------------------------------------------
# SHOW ALL OF THE FIGURES

plt.show()