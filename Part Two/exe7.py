# Neuromodulation Workshop: EXE 7
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
from skimage.measure import label, regionprops, regionprops_table
import plotly
import plotly.express as px
import plotly.graph_objects as go
from skimage import data, filters, measure, morphology
matplotlib.use('TkAgg')

# Read in image file
filename = 'NLE_s1_contra_GFAP-FITC_NeuN-CY5_20x_1.jpg'
image = plt.imread(filename)
M = np.asarray(image)

red_image = M[:, :, 0]      #R
green_image = M[:, :, 1]    #G
blue_image = M[:, :, 2]     #B

# Sharpen red channel
kernel_sharpen_1 = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
sharpened_image = cv2.filter2D(red_image, -1, kernel_sharpen_1)

# Apply a threshold to the sharpened image to create a mask
ret, thres_image = cv2.threshold(sharpened_image, 20, 255,cv2.THRESH_OTSU)

# Diamond function: https://stackoverflow.com/questions/58348401/numpy-array-filled-in-diamond-shape
def diamond(r):
    b = np.r_[:r, r:-1:-1]
    return (b[:, None]+b) >= r

# Close Image
close_mask = diamond(3)*1
close_mask = close_mask.astype(np.uint8)
closed_image = cv2.morphologyEx(thres_image, cv2.MORPH_CLOSE, close_mask, iterations=1)
# Open Image
open_mask = diamond(6)*1
open_mask = open_mask.astype(np.uint8)
opened_image = cv2.morphologyEx(closed_image, cv2.MORPH_OPEN, open_mask, iterations=1)

plt.figure()
plt.subplot(131)
plt.imshow(thres_image)
plt.title("Thresholded Image")
plt.subplot(132)
plt.imshow(closed_image)
plt.title("Closed Image")
plt.subplot(133)
plt.imshow(opened_image)
plt.title("Opened Image")

# Erode Image if regions appear connected
close_mask = diamond(2)*1
close_mask = close_mask.astype(np.uint8)
closed_image = cv2.morphologyEx(opened_image, cv2.MORPH_ERODE, close_mask, iterations=1)

# Label Image and
label_img = label(closed_image)

# Remove small specs from labelled image
cleaned_img = morphology.remove_small_objects(label_img, min_size=np.pi*5**2, connectivity=1)

# Find regions
regions = regionprops(cleaned_img)

# Visualise results from region props:
#   https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_regionprops.html
fig, ax = plt.subplots()
ax.imshow(opened_image, cmap=plt.cm.gray)
for props in regions:
    y0, x0 = props.centroid
    orientation = props.orientation
    x1 = x0 + math.cos(orientation) * 0.5 * props.minor_axis_length
    y1 = y0 - math.sin(orientation) * 0.5 * props.minor_axis_length
    x2 = x0 - math.sin(orientation) * 0.5 * props.major_axis_length
    y2 = y0 - math.cos(orientation) * 0.5 * props.major_axis_length

    ax.plot((x0, x1), (y0, y1), '-r', linewidth=2.5)
    ax.plot((x0, x2), (y0, y2), '-r', linewidth=2.5)
    ax.plot(x0, y0, '.g', markersize=15)


# Export regionprops data to excel spreadsheet
props = regionprops_table(label_img, properties=('centroid',
                                                 'area'))
# Save to a file
df = pd.DataFrame(props)
df.to_csv('cellProps.csv')

# Plot outlines on original image
fig = px.imshow(image, binary_string=True)
fig.update_traces(hoverinfo='skip') # hover is only for label info

numberOfCells = np.max(label_img)
print('Number of cells detected', numberOfCells)
# Find the average area
areaSum = 0
for props in regions:
    areaSum = areaSum+props.area
areaMean = areaSum/numberOfCells
print('Area average of cells:', areaMean)
print('Diameter average of cells:', np.sqrt(areaMean/np.pi)*2)

x_bins = np.linspace(0, np.max(df['area']), 100)
plt.figure()
plt.hist(df['area'], x_bins)
plt.title('Histogram plot of cell areas')

props = measure.regionprops(cleaned_img, image)

# For each label, add a filled scatter trace for its contour,
# and display the properties of the label in the hover of this trace.
for index in range(1, len(props)):
    label_i = props[index].label
    contour = measure.find_contours(cleaned_img == label_i, 0.5)[0]
    y, x = contour.T
    fig.add_trace(go.Scatter(
        x=x, y=y, name=label_i,
        mode='lines', fill='toself', showlegend=False, hoveron='points+fills'))

plotly.io.show(fig)
plt.show()
