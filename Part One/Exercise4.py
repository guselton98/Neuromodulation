# Neuromodulation Workshop: EXE 4
# Read from a file

import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(x, y)

# 2D Histogram plot
xedges = np.linspace(0, 10, num=10)
yedges = np.linspace(0, 10, num=10)
H, xedges, yedges = np.histogram2d(x, y, bins=(xedges, yedges))
H = H.T
fig = plt.figure(figsize=(7, 3))
ax = fig.add_subplot(131, title='imshow: square bins')
plt.imshow(H, interpolation='nearest', origin='lower',
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])