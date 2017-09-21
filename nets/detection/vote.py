import numpy as np
import math
import sys
import matplotlib.pyplot as plt
import pickle
import random
import cv2
import tensorflow as tf
from tqdm import tqdm
from sklearn.utils import shuffle
from skimage.exposure import equalize_hist
from tensorflow.contrib.layers import flatten
from scipy.misc import imread, imsave, imresize
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

data_file = "detection_data.p"

with open(data_file, mode='rb') as f:
    data = pickle.load(f)

X_train, y_train = data['xTrain'], data['yTrain']
X_valid, y_valid = data['xValidation'], data['yValidation']
X_test, y_test = data['xTest'], data['yTest']

all_labels = list(y_train) + list(y_valid) + list(y_test)
num_of_labels = np.unique(all_labels)

# Number of unique classes/labels in the dataset.
n_classes = len(num_of_labels)

#========PREPROCESSING==============
#===================================
def preprocess(X):
    X = np.reshape(X, (-1, 64, 64, 3))
    return X

X_test = preprocess(X_test)

def detect(model_name):
    gg = tf.Graph()
    with tf.Session(graph = gg) as sess:
        sess.run(tf.global_variables_initializer())
        saver2 = tf.train.import_meta_graph("./models/{}/model.meta".format(model_name))
        saver2.restore(sess, "./models/{}/model".format(model_name))

        prediction = gg.get_tensor_by_name("prediction:0")
        x = gg.get_tensor_by_name("input_data:0")
        keep_prob = gg.get_tensor_by_name("keep_prob:0")


        sess.run(prediction, feed_dict={x: X_test, keep_prob: 1.})

        predictions = (prediction.eval(feed_dict={x: X_test, keep_prob: 1.}))

    return predictions

final_detections = []
detections = {}
detections['model1'] = detect("detect-4")
detections['model2'] = detect("detect-6")
detections['model3'] = detect("detect-9")
detections['model4'] = detect("detect-9-1x1")
detections['model5'] = detect("detect-11")

for num in range(len(detections['model1'])):

    vote = 0
    for model in detections:
        vote += int(detections[model][num])

    if(vote >= (math.ceil(len(detections)/2))):
        final_detections.append(1)

    else:
        final_detections.append(0)

gg = tf.Graph()
with tf.Session(graph = gg) as sess:
    conf_mat = sess.run(tf.confusion_matrix(y_test, final_detections, n_classes))

    total = 0
    true_sum = 0
    false_sum = 0
    recalls = []
    precisions = []
    recall = 0
    precision = 0
    for i in range(len(conf_mat)): # row (actual)
        pred_pos = 0
        act_pos = 0
        for j in range(len(conf_mat[i])): # column of row (prediction)
            total += conf_mat[i][j]
            pred_pos += conf_mat[j][i]
            if(i == j):
                true_pos = conf_mat[i][j]
                true_sum += true_pos
                act_pos += true_pos
            elif(i != j):
                false_neg = conf_mat[i][j]
                false_sum += false_neg
                act_pos += false_neg
        if(act_pos == 0):
            act_pos = 1
        if(pred_pos == 0):
            pred_pos = 1
        recalls.append((true_pos/act_pos))
        precisions.append((true_pos/pred_pos))
    for i in range(len(recalls)):
        recall += recalls[i]
        precision += precisions[i]

    accuracy = (true_sum/total) * 100
    error_rate = (false_sum/total) * 100
    recall = (recall / len(recalls)) * 100
    precision = (precision / len(precisions)) * 100

    with open("detection_results.txt", mode='a') as f:
        f.write("Voting Results\n")
        f.write("---\n")
        f.write("Confusion matrix\n\n")
        f.write("Predicted\n {} <-- Actual\n".format(conf_mat))
        f.write("---\n")
        f.write("Error rate: {:.2f}%\n".format(error_rate))
        f.write("Recall: {:.2f}%\n".format(recall))
        f.write("Precision: {:.2f}%\n".format(precision))
        f.write("Network Accuracy: {:.2f}%\n".format(accuracy))
        f.write("------------------------------------\n")
        f.write("------------------------------------\n")
