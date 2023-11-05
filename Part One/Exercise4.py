# Neuromodulation Workshop: EXE 4
# Read from a file and perform some simple statistics

import pandas as pd
import numpy as np

data = pd.read_csv('cell_properties.csv')

print(data)
print(data.columns)
print(data['X Coor'])

# Store the data into separate variable (you don't have too, this is optional)
x = data['X Coor']
y = data['Y Coor']
area = data['Area']

# Find some statistics about the variables of the cells
print(x.mean())
print(np.mean(x))

# Other statistics
print(y.mean())
print(area.mean())

# try .min(), .max()
# Look at the list of mathematical functions that can be performed
#       https://numpy.org/doc/stable/reference/routines.math.html