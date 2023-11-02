import math
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import cv2

from skimage.measure import label, regionprops, regionprops_table

from skimage.transform import rotate
from matplotlib import cm
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)


import scipy.ndimage # useful resource aswell
from skimage.measure import label, regionprops, regionprops_table

from skimage.draw import ellipse
import skimage
# Tiff import/export
from PIL import Image

region_count = 2000

# Region 1
mu = 2
sigma = 1.2
region_1_count = 900
x1 = np.random.normal(mu, sigma, region_1_count)

# Region 2
mu = 5.5
sigma = 0.2
region_2_count = 500
x2 = np.random.normal(mu, sigma, region_2_count)

# Region 3
mu = 9
sigma = 0.8
region_3_count = 600
x3 = np.random.normal(mu, sigma, region_3_count)

x = np.concatenate((x1, x2, x3))
vec = np.linspace(0, 10, num=100)

plt.hist(x, bins='auto')

y = np.random.rand(region_count)

# Normalise coordinates between 0 to 10
x = (x-x.min())/(x.max()-x.min())*10
y = (y-y.min())/(y.max()-y.min())*10

# Assign random area for each circle (between 0.1 and 0.2 diameter)
area_of_regions = np.random.normal(0.15, 0.03, region_count)

# Create a data frame
df = pd.DataFrame({
    'X Coor': x,
    'Y Coor': y,
    'Area' : area_of_regions
})

# Export DF to CSV file
df.to_csv('cell_properties.csv')



#
#
#
#
#
# # Read in image file
# im = Image.open('../Pre-workshop/MAX_20x NeuN_2.tif')
# M = np.array(im)
# M = M/M.max()*255
# M = M.astype(np.uint8)
#
# ret, thresh = cv2.threshold(M,0,255,cv2.THRESH_OTSU)
#
# thresh = thresh - 255
# plt.imshow(thresh, 'gray')
# plt.show()
#
#
# # noise removal
# kernel = np.ones((3,3),np.uint8)
# #opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
# # sure background area
# sure_bg = cv2.dilate(thresh,kernel,iterations=3)
# # Finding sure foreground area
# dist_transform = cv2.distanceTransform(thresh,cv2.DIST_L2,3)
# ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
# plt.figure()
# plt.subplot(131)
# plt.imshow(thresh)
# plt.subplot(132)
# plt.imshow(sure_bg)
# plt.subplot(133)
# plt.imshow(sure_fg)
# plt.show()
#
# plt.figure()
# plt.imshow(dist_transform)
# plt.show()
#
# # Finding unknown region
# sure_fg = np.uint8(sure_fg)
# unknown = cv2.subtract(sure_bg,sure_fg)
# plt.figure()
# plt.subplot(131)
# plt.imshow(sure_bg)
# plt.subplot(132)
# plt.imshow(sure_fg)
# plt.subplot(133)
# plt.imshow(unknown)
# plt.show()
# # Marker labelling
# ret, markers = cv2.connectedComponents(sure_bg)
# # Add one to all labels so that sure background is not 0, but 1
# markers = markers+1
# # Now, mark the region of unknown with zero
# markers[unknown==255] = 0
# plt.imshow(markers)
# plt.show()
#
# r, c = M.shape
# RGB_im = np.zeros(shape=(r, c, 3))
# RGB_im[:, :, 1] = M
# RGB_im = RGB_im/RGB_im.max()*255
# RGB_im = RGB_im.astype(np.uint8)
#
# plt.imshow(RGB_im)
# plt.show()
# markers = cv2.watershed(RGB_im, markers)
# RGB_im[markers == -1] = [255,0,0]
# plt.imshow(RGB_im)
# plt.show()
#
# th3 = cv2.adaptiveThreshold(M, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
# th3 = 255-th3
# plt.imshow(th3, 'gray')
# plt.show()
#
# labelled_img = label(th3)
# regions = regionprops(labelled_img)
# plt.imshow(labelled_img)
# plt.show()
#
# def diamond(r):
#      b = np.r_[:r, r:-1:-1]
#      return (b[:, None]+b) >= r
#
# # Delete small specs
# spec_mask = diamond(1)*1
# spec_mask = spec_mask.astype(np.uint8)
# thres_image = cv2.morphologyEx(th3, cv2.MORPH_ERODE,
#                             spec_mask, iterations=1)
# thres_image = cv2.morphologyEx(thres_image, cv2.MORPH_DILATE,
#                             spec_mask, iterations=1)
# plt.imshow(thres_image, 'gray')
# plt.show()
#
#
# print(M)
# #th3 = cv2.adaptiveThreshold(M,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
#
# # Apply a Median offset gaussian blur kernel to the image
# def Gfilter(kernel_size, sigma=1, muu=0):
#     # Initializing value of x,y as grid of kernel size
#     # in the range of kernel size
#
#     x, y = np.meshgrid(np.linspace(-1, 1, kernel_size),
#                        np.linspace(-1, 1, kernel_size))
#     dst = np.sqrt(x ** 2 + y ** 2)
#     # lower normal part of gaussian
#     normal = 1 / (2 * np.pi * sigma ** 2)
#     # Calculating Gaussian filter
#     gauss = np.exp(-((dst - muu) ** 2 / (2.0 * sigma ** 2))) * normal
#     gauss = gauss - gauss[0, :].max()
#
#     return gauss
#
# #guassian_kernel = guassian_kernel-guassian_kernel[0,:].max()
# #guassian_kernel[guassian_kernel<0] = 0
# #guassian_kernel[guassian_kernel>0] = guassian_kernel[guassian_kernel>0] - guassian_kernel[guassian_kernel>0].mean()
#
# def filter_image(kernel_size, image):
#     gaussian_kernel = Gfilter(kernel_size)
#     filtered_image = cv2.filter2D(src=image, ddepth=-1, kernel=gaussian_kernel)
#     return filtered_image
#
# def update(kernel_size):
#     if kernel_size == 0:
#         return
#     image = M.copy()
#     GImage = filter_image(kernel_size, image)
#     cv2.imshow("image", GImage)
#     plt.imshow(GImage)
#     plt.show()
#     return GImage
#
# update(10)
# cv2.createTrackbar("Filtering Image...", "image", 4, 10,update)
# cv2.waitKey(0)

# M=M.astype(np.float32)
# guassian_kernel=guassian_kernel.astype(np.float32)
# filtered_image = cv2.filter2D(src=M, ddepth=-1, kernel=guassian_kernel)
# r, c = filtered_image.shape
# plt.imshow(filtered_image)
# plt.show()
#
# sub_image = filtered_image
# sub_image[sub_image>0] = 0
# plt.imshow(sub_image)
# plt.show()
#
#
#
#
# th3 = cv2.GaussianBlur(M,(5,5),0)
# th3 = cv2.adaptiveThreshold(th3,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,7,2)
# th3 = 255-th3
# plt.imshow(th3,'gray')
# plt.show()
# cv2.waitKey(0)
#
#
# shapes, hierarchy = cv2.findContours(th3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
# cv2.drawContours(image=M, contours=shapes, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
# for iteration, shape in enumerate(shapes):
#
#         if hierarchy[0,iteration,3] == -1:
#                 print(hierarchy[0,iteration,3])
#                 print(iteration)
#
# cv2.imshow('Shapes', M)
# cv2.waitKey(0)
#
#
# level = 10000
# cv2.drawContours(M, contours, level, (128, 255, 255), 1)
# cv2.imshow('Contours', M)
#
# h, w = M.shape[:2]
#
# def update(levels):
#     vis = np.zeros((h, w, 3), np.uint8)
#     levels = levels - 3
#     cv2.drawContours(vis, contours, (-1, 3)[levels <= 0], (128, 255, 255), 3, cv2.CV_8S, hierarchy, abs(levels))
#     cv2.imshow('contours', vis)
#
# update(3)
# cv2.createTrackbar("levels+3", "contours", 3, 7, update)
# cv2.imshow('image', M)
#
#
# # Plot separate RGB channels of image
# # Sharpen red channel
# #plt.figure(figsize=(12, 6))
# #kernel_sharpen_1 = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
# #sharpened_image = cv2.filter2D(M, -1, kernel_sharpen_1)
# #plt.imshow(sharpened_image)
# #plt.title("Sharpened Image")
#
# # Apply a threshold to the sharpened image to create a mask
# ret, thres_image = cv2.threshold(M, 40, 255,cv2.THRESH_BINARY)
#
# def diamond(r):
#     b = np.r_[:r, r:-1:-1]
#     return (b[:, None]+b) >= r
#
# # Delete small specs
# spec_mask = diamond(5)*1
# spec_mask = spec_mask.astype(np.uint8)
# thres_image = cv2.morphologyEx(thres_image, cv2.MORPH_OPEN,
#                            spec_mask, iterations=1)
# thres_image = cv2.morphologyEx(thres_image, cv2.MORPH_CLOSE,
#                            spec_mask, iterations=1)
#
# # Close Image
# close_mask = diamond(20)*1
# close_mask = close_mask.astype(np.uint8)
# closed_image = cv2.morphologyEx(thres_image, cv2.MORPH_CLOSE,
#                            close_mask, iterations=1)
# # Open Image
# open_mask = diamond(10)*1
# open_mask = open_mask.astype(np.uint8)
# opened_image = cv2.morphologyEx(closed_image, cv2.MORPH_OPEN,
#                            open_mask, iterations=1)
#
# # Plot the results of opening and closing a thresholded image
# plt.figure()
# plt.subplot(131)
# plt.imshow(thres_image)
# plt.title("Thresholded Image")
# plt.subplot(132)
# plt.imshow(closed_image)
# plt.title("Closed Image")
# plt.subplot(133)
# plt.imshow(opened_image)
# plt.title("Opened Image")
#
# # Overlay opened_image with red channel
#
# plt.figure()
# plt.subplot(121)
# plt.imshow(opened_image, cmap='Reds', vmin=0, vmax=255)
# plt.title('Final Image')
# plt.subplot(122)
# plt.imshow(image)
# plt.title("Original Image")
#
# # Apply mask to image (multiply them together)
# final = opened_image*image
# plt.figure()
# plt.imshow(final)
# plt.title("Multiplied")
#
# def detect_peaks(I):
#     # https://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array
#     """
#     Takes an image and detect the peaks usingthe local maximum filter.
#     Returns a boolean mask of the peaks (i.e. 1 when
#     the pixel's value is the neighborhood maximum, 0 otherwise)
#     """
#     # define an 8-connected neighborhood
#     neighborhood = generate_binary_structure(2,10)
#     #apply the local maximum filter; all pixel of maximal value
#     #in their neighborhood are set to 1
#     local_max = maximum_filter(I, footprint=neighborhood)==I
#     #local_max is a mask that contains the peaks we are
#     #looking for, but also the background.
#     #In order to isolate the peaks we must remove the background from the mask.
#     #we create the mask of the background
#     background = (I==0)
#     #a little technicality: we must erode the background in order to
#     #successfully subtract it form local_max, otherwise a line will
#     #appear along the background border (artifact of the local maximum filter)
#     eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)
#     #we obtain the final mask, containing only peaks,
#     #by removing the background from the local_max mask (xor operation)
#     detected_peaks = local_max ^ eroded_background
#     return detected_peaks
#
# # Generate mesh for point cloud
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# [R, C] = final.shape
# X = np.arange(0, R, 1)
# Y = np.arange(0, C, 1)
# X, Y = np.meshgrid(Y, X)
# plt.figure()
# surf = ax.plot_surface(X, Y, M, cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)
#
# # Denoising
# final = final/final.max()*255
# final = final.astype(np.uint8)
#
#
#
#
# peaks = detect_peaks(denoise_bilateral(final))
# plt.figure()
# plt.imshow(peaks)
# plt.imshow(denoise_bilateral(final))
# plt.show()
#
# # Label Image and find region properties
# label_img = label(opened_image)
# regions = regionprops(label_img)
#
# # Visualise results from regionprops
# plt.figure()
# fig, ax = plt.subplots()
# ax.imshow(opened_image, cmap=plt.cm.gray)
# for props in regions:
#     y0, x0 = props.centroid
#     orientation = props.orientation
#     x1 = x0 + math.cos(orientation) * 0.5 * props.minor_axis_length
#     y1 = y0 - math.sin(orientation) * 0.5 * props.minor_axis_length
#     x2 = x0 - math.sin(orientation) * 0.5 * props.major_axis_length
#     y2 = y0 - math.cos(orientation) * 0.5 * props.major_axis_length
#
#     ax.plot((x0, x1), (y0, y1), '-r', linewidth=2.5)
#     ax.plot((x0, x2), (y0, y2), '-r', linewidth=2.5)
#     ax.plot(x0, y0, '.g', markersize=15)
# plt.show()
#
# # Overlay with Red image
#
# # Overlay with original Image
#
#
# # Export regionprops data to excel spreadsheet
# props = regionprops_table(label_img, properties=('centroid',
#                                                  'orientation',
#                                                  'solidity'))
# df = pd.DataFrame(props)
# df.to_csv('Part Two\\temp.csv')