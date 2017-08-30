import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import random
import time
import datetime
from tqdm import tqdm
from collections import deque
from skimage.feature import hog
from sklearn.model_selection import train_test_split, GridSearchCV
from scipy.misc import imread, imresize
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from scipy.ndimage.measurements import label
from skimage.exposure import equalize_hist
from skimage.io import imsave

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

IMG_SIZE = 64


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
        # make a new image of all zero values (so a black image) the same size as img
        heatmap = np.zeros_like(img[:,:,0])

        vehicleBoxes = []
        # Make a copy of the image
        imcopy = np.copy(img)
        images = []
        # Iterate through the bounding boxes
        for bbox in bboxes:
            startx = bbox[0][0]
            endx = bbox[1][0]
            starty = bbox[0][1]
            endy = bbox[1][1]
            images.append(imcopy[starty:endy, startx:endx])

        predictions = self.predict(images)
        for num in range(len(predictions)):
            startx = bboxes[num][0][0]
            endx = bboxes[num][1][0]
            starty = bboxes[num][0][1]
            endy = bboxes[num][1][1]

            if(predictions[num] == 1):
                cv2.rectangle(imcopy, (startx, starty), (endx, endy), color, thick)
                heatmap[starty:endy, startx:endx] += 1
        # Return the image copy with boxes drawn
        return imcopy, heatmap


    def add_heat(heatmap, bbox_list):
        # Iterate through list of bboxes
        for box in bbox_list:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

        # Return updated heatmap
        return heatmap


    def predict(self, images):

        my_images = []

        for image in images:
            new_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (IMG_SIZE,IMG_SIZE))
            # # for c in range(3):
            # #     image[:,:,c] = image[:,:,c] - np.mean(image[:,:,c])
            # gray = cv2.cvtColor(new_image, cv2.COLOR_RGB2GRAY)
            # blur = cv2.GaussianBlur(gray, (5,5), 20.0)
            # image = cv2.addWeighted(gray, 2, blur, -1, 0)
            # image = cv2.equalizeHist(image)
            # image = equalize_hist(image)
            my_images.append(image)

        my_images = np.asarray(my_images)
        my_images = np.reshape(my_images, (-1, IMG_SIZE, IMG_SIZE, 3))

        print("predicting {} images...".format(len(images)))
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph('color_net/model.meta')
            saver.restore(sess, tf.train.latest_checkpoint('color_net/.'))
            graph = tf.get_default_graph()
            #print("Model restored...")
            x = graph.get_tensor_by_name("input_data:0")
            #print("x restored...")
            keep_prob = graph.get_tensor_by_name("keep_prob:0")
            #print("keep_prob...")
            prediction = graph.get_tensor_by_name("prediction:0")
            #print("prediction op restored...")


            sess.run(prediction, feed_dict={x: my_images, keep_prob: 1.})
            #print("prediction op finished...")
            predictions = (prediction.eval(feed_dict={x: my_images, keep_prob: 1.}))
            #print("predictions assigned...")
        print("DONE PREDICTING")
        return predictions


# scan = Scanner()
# for i in tqdm(range(10)):
#     img = imread("test_imgs/test{}.jpg".format(i+1))
#     windows = scan.slide_window(img, x_start_stop=[None, None], y_start_stop=[None, 2300], xy_window=(64, 64), xy_overlap=(0.5, 0.5))
#     window_img = scan.draw_boxes(img, windows, color=(0, 0, 255), thick=8)
#     now = datetime.datetime.now()
#     d = now.day
#     m = now.minute
#     s = now.second
#     imsave("outputImages/final_img_{}.jpg".format(i+1), window_img)

scan = Scanner()
img = imread("test_imgs/test10.jpg")
windows = scan.slide_window(img, x_start_stop=[None, None], y_start_stop=[500, 2300], xy_window=(64, 64), xy_overlap=(0.5, 0.5))
window_img, heatmap = scan.draw_boxes(img, windows, color=(0, 0, 255), thick=8)
fig = plt.figure()
plt.subplot(121)
plt.imshow(window_img)
plt.title('Sign Position')
plt.subplot(122)
plt.imshow(heatmap, cmap='hot')
plt.title('Heat Map')
fig.tight_layout()
plt.show()
