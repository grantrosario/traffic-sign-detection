import numpy as np
import pickle
import cv2
import tensorflow as tf
from sklearn.utils import shuffle
from skimage.exposure import equalize_hist
from tensorflow.contrib.layers import flatten
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

data_file = "recognition_data.p"

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
    t = []
    for i in range(0, len(X)):
        gray = cv2.cvtColor(X[i], cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 20.0)
        image = cv2.addWeighted(gray, 2, blur, -1, 0)
        image = cv2.equalizeHist(image)
        image = equalize_hist(image)
        t.append(image)
    X = np.reshape(t, (-1, 64, 64, 1))
    return X

X_test = preprocess(X_test)

def recognize(model_name):
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

final_recognitions = []
xer = []
recognitions = {}
recognitions['model1'] = recognize("transfer_recognize-4")
recognitions['model2'] = recognize("transfer_recognize-6")
recognitions['model3'] = recognize("transfer_recognize-9")
recognitions['model4'] = recognize("transfer_recognize-9-1x1")
recognitions['model5'] = recognize("transfer_recognize-11")

for num in range(len(recognitions['model1'])):
    vote = {}
    for model in recognitions:
        vote[str(recognitions[model][num])] = 0
    for model in recognitions:
        vote[str(recognitions[model][num])] += 1

    max_vote = max(vote.values())
    winner = [k for k, v in vote.items() if v == max_vote]
    final_recognitions.append(int(winner[0]))


gg = tf.Graph()
with tf.Session(graph = gg) as sess:
    conf_mat = sess.run(tf.confusion_matrix(y_test, final_recognitions, 43))

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

    with open("transfer_recognition_results.txt", mode='a') as f:
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
