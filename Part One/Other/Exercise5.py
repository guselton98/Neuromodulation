# Neuromodulation Workshop: EXE 5
# Read from a file and plot some stuff

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

# Plot a map of the cells using a scatter plot
fig, ax = plt.subplots()
ax.scatter(x, y, marker='o')

# Add a title and x and y label
ax.set(title="Plot of cell locations recorded",
       xlabel="Longitude",
       ylabel="Latitude")

# USe matplotlib.pyplot.show() to make the figure appear
plt.show()

# What do you notice??
