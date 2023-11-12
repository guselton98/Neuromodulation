# NEUROMODULATION WORKSHOP
# PART 2 - EXERCISE 2: CSV FILE IMPORT AND STATISTICAL ANALYSIS

#----------------------------------------------------------------------------------------------------------------------
# IMPORTING LIBRARIES

# Renaming the reference name, so we don't need to constantly type out long names in our code.
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
#----------------------------------------------------------------------------------------------------------------------
# READ IN FILE

data = pd.read_csv('cell_properties.csv')

#----------------------------------------------------------------------------------------------------------------------
# STORE INFORMATION

print(data.columns)   # Print the titles of the columns, so we know what to reference.

# Store the data into separate variables, using those column titles.
x = data['X Coor']
y = data['Y Coor']
area = data['Area']

#----------------------------------------------------------------------------------------------------------------------
# STATISTICAL ANALYSIS
# Find some statistics about the variables of the cells.

# NB: This syntax looks similar, but they do vastly different things:
print(x.mean())   # Applying the Python in-built function mean() to the data stored in our object named 'x'.
print(np.mean(x))   # Using the mean function from the numpy library and passing the stored data within 'x' into it.

# Same deal as above, but using 'y':
print(y.mean())
print(area.mean())

# Try min() and max() functions
# Look at the list of mathematical functions that can be performed
# https://numpy.org/doc/stable/reference/routines.math.html
# NB: It's a good habit to fact-check your answers through manual calculation

#----------------------------------------------------------------------------------------------------------------------