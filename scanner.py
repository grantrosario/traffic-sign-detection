import numpy as np
import math
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import glob
import random
import time
import csv
import os
import datetime
from tqdm import tqdm
from scipy.misc import imread
from scipy.ndimage.measurements import label
from skimage.exposure import equalize_hist
from skimage.io import imsave

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

IMG_SIZE = 64

class Scanner():

    def __init__(self):
        self.g = tf.Graph()
        self.h = tf.Graph()

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
        predictions = {}
        # Iterate through the bounding boxes
        for bbox in bboxes:
            startx = bbox[0][0]
            endx = bbox[1][0]
            starty = bbox[0][1]
            endy = bbox[1][1]
            images.append(imcopy[starty:endy, startx:endx])

        print("predicting {} images...".format(len(images)))
        predictions['model1'] = self.detect(images, 'models/detect-4/', tf.Graph())
        predictions['model2'] = self.detect(images, 'models/detect-6/', tf.Graph())
        predictions['model3'] = self.detect(images, 'models/detect-9/', tf.Graph())
        predictions['model4'] = self.detect(images, 'models/detect-9-1x1/', tf.Graph())
        predictions['model5'] = self.detect(images, 'models/detect-11/', tf.Graph())
        for num in range(len(predictions['model1'])):
            startx = bboxes[num][0][0]
            endx = bboxes[num][1][0]
            starty = bboxes[num][0][1]
            endy = bboxes[num][1][1]

            vote = 0
            for model in predictions:
                vote += int(predictions[model][num])

            if(vote >= (math.ceil(len(predictions)/2))):
                cv2.rectangle(imcopy, (startx, starty), (endx, endy), color, thick)
                heatmap[starty:endy, startx:endx] += 1

        # Return the image copy with boxes drawn
        return imcopy, heatmap

    def apply_threshold(self, heatmap, threshold):
        heatmap[heatmap <= threshold] = 0
        return heatmap

    def show_image_search(self, img, bboxes):
        imcopy = np.copy(img)
        for num in range(len(bboxes)):
            startx = bboxes[num][0][0]
            endx = bboxes[num][1][0]
            starty = bboxes[num][0][1]
            endy = bboxes[num][1][1]
            cv2.rectangle(imcopy, (startx, starty), (endx, endy), (0,0,255), 5)
        plt.imshow(imcopy)
        plt.show()

    def draw_labeled_boxes(self, img, labels):
        signs = []
        textboxes = []
        for sign_num in range(1, labels[1]+1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == sign_num).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            startx = np.min(nonzerox) - 30
            endx = np.max(nonzerox) + 30
            starty = np.min(nonzeroy) - 35
            endy = np.max(nonzeroy) + 30
            # Define a bounding box based on min/max x and y
            bbox = ((startx, starty), (endx, endy))
            # append picture of sign for future recognition
            signs.append(img[starty:endy, startx:endx])
            # append position for bottom left corner of text box above signs
            textboxes.append((startx, endy+30))
            # Draw the box on the image
            cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
        # Return the image
        return img, signs, textboxes


    def detect(self, images, model_path, graph):

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

        with tf.Session(graph = graph) as sess:
            saver = tf.train.import_meta_graph('{}model.meta'.format(model_path))
            saver.restore(sess, tf.train.latest_checkpoint('{}.'.format(model_path)))
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
        return predictions

    def recognize(self, images, model_path, graph):

        my_images = []

        for image in images:
            new_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(new_image, (IMG_SIZE,IMG_SIZE))
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            blur = cv2.GaussianBlur(gray, (5,5), 20.0)
            image = cv2.addWeighted(gray, 2, blur, -1, 0)
            image = cv2.equalizeHist(image)
            image = equalize_hist(image)
            my_images.append(image)

        my_images = np.asarray(my_images)
        my_images = np.reshape(my_images, (-1, IMG_SIZE, IMG_SIZE, 1))

        with tf.Session(graph = graph) as sess:
            saver = tf.train.import_meta_graph('{}model.meta'.format(model_path))
            saver.restore(sess, tf.train.latest_checkpoint('{}.'.format(model_path)))

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
        return predictions

    def get_recognitions(self, signs):
        final_recognitions = []
        recognitions = {}
        print("recognizing {} sign(s)...".format(len(signs)))
        recognitions['model1'] = scan.recognize(signs, "models/recognize-4/", tf.Graph())
        recognitions['model2'] = scan.recognize(signs, "models/recognize-6/", tf.Graph())
        recognitions['model3'] = scan.recognize(signs, "models/recognize-9/", tf.Graph())
        recognitions['model4'] = scan.recognize(signs, "models/recognize-9-1x1/", tf.Graph())
        recognitions['model5'] = scan.recognize(signs, "models/recognize-11/", tf.Graph())

        for num in range(len(recognitions['model1'])):
            vote = {}
            for model in recognitions:
                vote[str(recognitions[model][num])] = 0
            for model in recognitions:
                vote[str(recognitions[model][num])] += 1

            max_vote = max(vote.values())
            winner = [k for k, v in vote.items() if v == max_vote]
            final_recognitions.append(winner[0])
        return final_recognitions

    def draw_sign(self, img, textboxes, recognitions):
        sign_truths = {}
        with open('signLabels.csv') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                sign_truths[row['label']] = row['name']
        font = cv2.FONT_HERSHEY_DUPLEX
        for idx in range(0, len(textboxes)):
            text = "{}".format(sign_truths[str(recognitions[idx])])
            cv2.putText(img, text, textboxes[idx], font, 2, (100, 255, 100), 2, cv2.LINE_AA)
            print("sign {} recognized as {}".format(idx+1, sign_truths[str(recognitions[idx])]))




# scan = Scanner()
# img = cv2.imread("test_imgs/test1.jpg")
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img = cv2.resize(img, (4032, 3024))
# windows = scan.slide_window(img, x_start_stop=[None, None], y_start_stop=[1100, 1500], xy_window=(64, 64), xy_overlap=(0.5, 0.5))
# windows += scan.slide_window(img, x_start_stop=[None, None], y_start_stop=[900, 1700], xy_window=(128, 128), xy_overlap=(0.5, 0.5))
# windows += scan.slide_window(img, x_start_stop=[None, None], y_start_stop=[700, 1900], xy_window=(256, 256), xy_overlap=(0.5, 0.5))
# windows += scan.slide_window(img, x_start_stop=[None, None], y_start_stop=[500, 2100], xy_window=(512, 512), xy_overlap=(0.5, 0.5))
# scan.show_image_search(img, windows)
# for i in tqdm(range(10)):
#     img = cv2.imread("test_imgs/test{}.jpg".format(i+1))
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = cv2.resize(img, (4032, 3024))
#     windows = scan.slide_window(img, x_start_stop=[None, None], y_start_stop=[1100, 1500], xy_window=(64, 64), xy_overlap=(0.5, 0.5))
#     windows += scan.slide_window(img, x_start_stop=[None, None], y_start_stop=[900, 1700], xy_window=(128, 128), xy_overlap=(0.5, 0.5))
#     windows += scan.slide_window(img, x_start_stop=[None, None], y_start_stop=[700, 1900], xy_window=(256, 256), xy_overlap=(0.5, 0.5))
#     windows += scan.slide_window(img, x_start_stop=[None, None], y_start_stop=[500, 2100], xy_window=(512, 512), xy_overlap=(0.5, 0.5))
#     window_img, heat = scan.draw_boxes(img, windows, color=(0, 0, 255), thick=8)
#     now = datetime.datetime.now()
#     d = now.day
#     m = now.minute
#     s = now.second
#     imsave("outputImages/window_img_{}.jpg".format(i+1), window_img)



# # 1, 3, 4, 5
scan = Scanner()
# img = Image.open("test_imgs/test3.jpg")
# img = img.rotate(180)
# img.save("test_imgs/test3.jpg")
img = cv2.imread("test_imgs/test2.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (4032, 3024))

windows = scan.slide_window(img, x_start_stop=[None, None], y_start_stop=[1100, 1500], xy_window=(64, 64), xy_overlap=(0.5, 0.5))
windows += scan.slide_window(img, x_start_stop=[None, None], y_start_stop=[900, 1700], xy_window=(128, 128), xy_overlap=(0.5, 0.5))
windows += scan.slide_window(img, x_start_stop=[None, None], y_start_stop=[700, 1900], xy_window=(256, 256), xy_overlap=(0.5, 0.5))
windows += scan.slide_window(img, x_start_stop=[None, None], y_start_stop=[500, 2100], xy_window=(512, 512), xy_overlap=(0.5, 0.5))
window_img, heatmap = scan.draw_boxes(img, windows, color=(0, 0, 255), thick=8)

heat2 = scan.apply_threshold(heatmap, 2)

labels = label(heat2)

# plt.imshow(labels[0])
# plt.show()

draw_img, signs, textboxes = scan.draw_labeled_boxes(np.copy(img), labels)

recognitions = scan.get_recognitions(signs)

scan.draw_sign(draw_img, textboxes, recognitions)
plt.imshow(draw_img)
plt.title(recognitions[0])
plt.show()
