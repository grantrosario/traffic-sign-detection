import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import random
import time
import tensorflow as tf
from collections import deque
from skimage.feature import hog
from sklearn.model_selection import train_test_split, GridSearchCV
from scipy.misc import imread, imsave, imresize
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from scipy.ndimage.measurements import label
from skimage.exposure import equalize_hist


class Scanner():

    def slide_window(self, img, x_start_stop=[None, None], y_start_stop=[None, None],
                     xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
        # If x and/or y start/stop positions not defined, set to image size
        if x_start_stop[0] == None:
            x_start_stop[0] = 0
        if x_start_stop[1] == None:
            x_start_stop[1] = img.shape[1]
        if y_start_stop[0] == None:
            y_start_stop[0] = 0
        if y_start_stop[1] == None:
            y_start_stop[1] = img.shape[0]

        # Compute the span of the region to be searched
        xspan = x_start_stop[1] - x_start_stop[0]
        yspan = y_start_stop[1] - y_start_stop[0]

        # Compute the number of pixels per step in x/y
        nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
        ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))

        # Compute the number of windows in x/y
        nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
        ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
        nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step)
        ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step)

         # Initialize a list to append window positions to
        window_list = []

        # Loop through finding x and y window positions
        #     Note: you could vectorize this step, but in practice
        #     you'll be considering windows one by one with your
        #     classifier, so looping makes sense
        for ys in range(ny_windows):
            for xs in range(nx_windows):
                # Calculate window position
                startx = xs*nx_pix_per_step + x_start_stop[0]
                endx = startx + xy_window[0]
                starty = ys*ny_pix_per_step + y_start_stop[0]
                endy = starty + xy_window[1]
                # Append window position to list
                window_list.append(((startx, starty), (endx, endy)))
        # Return the list of windows
        return window_list

    def draw_boxes(self, img, bboxes, color=(0, 0, 255), thick=6):
        # Make a copy of the image
        imcopy = np.copy(img)
        # Iterate through the bounding boxes
        for bbox in bboxes:
            # Draw a rectangle given bbox coordinates
            cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
        # Return the image copy with boxes drawn
        return imcopy



scan = Scanner()
img = imread("test_imgs/test1.jpg")
windows = scan.slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], xy_window=(128, 128), xy_overlap=(0.5, 0.5))
window_img = scan.draw_boxes(img, windows, color=(0, 0, 255), thick=6)

# plt.imshow(window_img)
# plt.imshow(crp)
# plt.show()

my_images = []
crp = img[1570:1750, 1590:1740]
# plt.imshow(crp)
# plt.show()

# from model import *
# image = cv2.cvtColor(crp, cv2.COLOR_BGR2RGB)
#
# gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
# blur = cv2.GaussianBlur(gray, (5,5), 20.0)
# image = cv2.addWeighted(gray, 2, blur, -1, 0)
# image = cv2.equalizeHist(image)
# image = equalize_hist(image)
# sized = cv2.resize(image, (64,64))
# my_images.append(sized)
#
# my_images = np.asarray(my_images)
# my_images = np.reshape(my_images, (-1, 64, 64, 1))
# my_labels = [0]
#
# with tf.Session() as sess:
#     saver.restore(sess, tf.train.latest_checkpoint('nets/.'))
#
#     my_accuracy = evaluate(my_images, my_labels)
#     print("\n\nMy Accuracy = {:.3f}".format(my_accuracy))
