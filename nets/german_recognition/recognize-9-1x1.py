import numpy as np
import sys
import matplotlib.pyplot as plt
import pickle
import random
import cv2
from tqdm import tqdm
from sklearn.utils import shuffle
from skimage.exposure import equalize_hist
from tensorflow.contrib.layers import flatten
from scipy.misc import imread, imsave, imresize

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

# np.set_printoptions(threshold=sys.maxsize)

data_file = "german_recognition_data.p"

with open(data_file, mode='rb') as f:
    data = pickle.load(f)

X_train, y_train = data['xTrain'], data['yTrain']
X_valid, y_valid = data['xValidation'], data['yValidation']
X_test, y_test = data['xTest'], data['yTest']

all_labels = list(y_train) + list(y_valid) + list(y_test)
num_of_labels = np.unique(all_labels)

# Number of training examples
n_train = len(X_train)

# Number of testing examples
n_test = len(X_test)

# Shape of a traffic sign image
image_shape = (X_train[0].shape)

# Number of unique classes/labels in the dataset.
n_classes = len(num_of_labels)

#========VISUALIZATION==============
#===================================

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

# generate 5 random data points and show images
fig, axs = plt.subplots(1, 5, figsize=(15, 6))  # create plot boxes for images
fig.subplots_adjust(hspace = .2, wspace=.1)     # adjust height and width of spacing around boxes
# axs = axs.ravel(order='C')                      # flatten array into 1-D array
for i in range(5):
    index = random.randint(0, len(X_train))
    image = X_train[index]
    axs[i].axis('off')
    axs[i].imshow(image)
    axs[i].set_title(y_train[index])
#plt.show()

#Plot histogram of training labels used
plt.figure(figsize=(12, 4))
hist, bins = np.histogram(y_train, bins = n_classes)
width = 0.7 * (bins[1] - bins[0]) / 2
center = (bins[:-1] + bins[1:]) / 2
barlist = plt.bar(center, hist, align = 'center', width=width, color='royalblue')
plt.title("Frequency of labels used")
plt.xlabel("Label number")
plt.ylabel("Number of images")
#plt.show()

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
    print("Image Shape: {}".format(X.shape))
    return X


print("Training data")
# show raw image vs processed image
# X_original = X_train
# X_processed = preprocess(X_train)
# fig, axs = plt.subplots(1,2, figsize=(10, 3))
# axs = axs.ravel()
#
# axs[0].axis('off')
# axs[0].set_title('Original')
# axs[0].imshow(X_original[0].squeeze())
#
# axs[1].axis('off')
# axs[1].set_title('Processed')
# axs[1].imshow(X_processed[0].squeeze(), cmap='gray')


X_train = preprocess(X_train)
print("Validation data")
X_valid = preprocess(X_valid)
print("Test data")
X_test = preprocess(X_test)
X_train, y_train = shuffle(X_train, y_train)


### Define your architecture here.
### Feel free to use as many code cells as needed.
import tensorflow as tf
from tensorflow.contrib.layers import flatten

EPOCHS = 100
BATCH_SIZE = 64
beta = 0.001

def LeNet(x):
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    ft_sz = 3

    # TODO: Layer 1: Convolutional. Input = 64x64x1. Output = 64x64x8.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(ft_sz,ft_sz,1,8), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(8))
    conv1   = tf.nn.conv2d(x, conv1_W, strides = [1, 1, 1, 1], padding = 'SAME') + conv1_b
    regularizers = tf.nn.l2_loss(conv1_W)
    # TODO: Activation.
    conv1 = tf.nn.relu(conv1)

    # TODO: Pooling. Input = 64x64x8. Output = 32x32x8.
    conv1 = tf.nn.max_pool(conv1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

    # TODO: Layer 2: Convolutional. Output = 32x32x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(ft_sz, ft_sz, 8, 16), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides = [1, 1, 1, 1], padding = 'SAME') + conv2_b
    regularizers += tf.nn.l2_loss(conv2_W)
    # TODO: Activation.
    conv2 = tf.nn.relu(conv2)

    # TODO: Pooling. Input = 32x32x16. Output = 16x16x16.
    conv2 = tf.nn.max_pool(conv2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

    # TODO: Layer 2: Convolutional. Output = 16x16x32.
    conv3_W = tf.Variable(tf.truncated_normal(shape=(ft_sz, ft_sz, 16, 32), mean = mu, stddev = sigma))
    conv3_b = tf.Variable(tf.zeros(32))
    conv3   = tf.nn.conv2d(conv2, conv3_W, strides = [1, 1, 1, 1], padding = 'SAME') + conv3_b
    regularizers += tf.nn.l2_loss(conv3_W)
    # TODO: Activation.
    conv3 = tf.nn.relu(conv3)

    # TODO: Layer 1: Convolutional. Input = 16x16x16. Output = 16x16x32.
    conv3_2_W = tf.Variable(tf.truncated_normal(shape=(1,1,32,32), mean = mu, stddev = sigma))
    conv3_2_b = tf.Variable(tf.zeros(32))
    conv3   = tf.nn.conv2d(conv3, conv3_2_W, strides = [1, 1, 1, 1], padding = 'SAME') + conv3_2_b
    regularizers = tf.nn.l2_loss(conv3_2_W)
    # TODO: Activation.
    conv3 = tf.nn.relu(conv3)

    # TODO: Pooling. Input = 16x16x32. Output = 8x8x32.
    conv3 = tf.nn.max_pool(conv3, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

    # TODO: Layer 2: Convolutional. Output = 8x8x64.
    conv4_W = tf.Variable(tf.truncated_normal(shape=(ft_sz, ft_sz, 32, 64), mean = mu, stddev = sigma))
    conv4_b = tf.Variable(tf.zeros(64))
    conv4   = tf.nn.conv2d(conv3, conv4_W, strides = [1, 1, 1, 1], padding = 'SAME') + conv4_b
    regularizers += tf.nn.l2_loss(conv4_W)
    # TODO: Activation.
    conv4 = tf.nn.relu(conv4)

    # TODO: Layer 1: Convolutional. Input = 8x8x32. Output = 8x8x64.
    conv4_2_W = tf.Variable(tf.truncated_normal(shape=(1,1,64,64), mean = mu, stddev = sigma))
    conv4_2_b = tf.Variable(tf.zeros(64))
    conv4   = tf.nn.conv2d(conv4, conv4_2_W, strides = [1, 1, 1, 1], padding = 'SAME') + conv4_2_b
    regularizers = tf.nn.l2_loss(conv4_2_W)
    # TODO: Activation.
    conv4 = tf.nn.relu(conv4)

    # TODO: Pooling. Input = 8x8x64. Output = 4x4x64.
    conv4 = tf.nn.max_pool(conv4, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

    # TODO: Layer 2: Convolutional. Output = 4x4x128.
    conv5_W = tf.Variable(tf.truncated_normal(shape=(ft_sz, ft_sz, 64, 128), mean = mu, stddev = sigma))
    conv5_b = tf.Variable(tf.zeros(128))
    conv5   = tf.nn.conv2d(conv4, conv5_W, strides = [1, 1, 1, 1], padding = 'SAME') + conv5_b
    regularizers += tf.nn.l2_loss(conv5_W)
    # TODO: Activation.
    conv5 = tf.nn.relu(conv5)

    # TODO: Layer 1: Convolutional. Input = 4x4x64. Output = 4x4x128.
    conv5_2_W = tf.Variable(tf.truncated_normal(shape=(1,1,128,128), mean = mu, stddev = sigma))
    conv5_2_b = tf.Variable(tf.zeros(128))
    conv5   = tf.nn.conv2d(conv5, conv5_2_W, strides = [1, 1, 1, 1], padding = 'SAME') + conv5_2_b
    regularizers = tf.nn.l2_loss(conv5_2_W)
    # TODO: Activation.
    conv5 = tf.nn.relu(conv5)

    # TODO: Pooling. Input = 4x4x128. Output = 2x2x128.
    conv5 = tf.nn.max_pool(conv5, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

    # TODO: Flatten. Input = 2x2x128. Output = 512.
    fc0   = flatten(conv5)

    # TODO: Layer 3: Fully Connected. Input = 2048. Output = 120.
    fc1_W  = tf.Variable(tf.truncated_normal(shape=(512, 43), mean = mu, stddev = sigma))
    fc1_b  = tf.Variable(tf.zeros(43))
    regularizers += tf.nn.l2_loss(fc1_W)
    logits = tf.matmul(fc0, fc1_W) + fc1_b

    return [logits, regularizers]


rate = 0.001
keep_prob = tf.placeholder(tf.float32, name="keep_prob") # probablity of keeping for dropout
x = tf.placeholder(tf.float32, (None, 64, 64, 1), name="input_data")
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 43)

logits, regularizers = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
loss_operation = tf.reduce_mean(loss_operation + beta * regularizers) # L2 regularization
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

prediction = tf.argmax(logits, 1, name="prediction")
correct_prediction = tf.equal(prediction, tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

#=============EVALUATION============
#===================================
def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in tqdm(range(0, num_examples, BATCH_SIZE)):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


#=============TRAINING==============
#===================================
### Train the model.
### Calculate and report the accuracy on the training and validation set.
### FEATURES AND LABELS
if((input('Would you like to train? (y/n): ')) == 'y'):

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        num_examples = len(X_train)

        prev_val_acc = 0
        early_stop_counter = 0
        rate_decay = 0.0001
        print()
        print("Training...")
        for i in range(EPOCHS):
            X_train, y_train = shuffle(X_train, y_train)
            for offset in tqdm(range(0, num_examples, BATCH_SIZE)):
                end = offset + BATCH_SIZE
                batch_x, batch_y = X_train[offset:end], y_train[offset:end]
                sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})

            print()
            print("Evaluating accuracy...")
            validation_accuracy = evaluate(X_valid, y_valid)
            training_accuracy = evaluate(X_train, y_train)
            print("EPOCH {} ...".format(i+1))
            print("Validation Accuracy = {:.3f}".format(validation_accuracy))
            print("Training Accuracy = {:.3f}".format(training_accuracy))

            #EARLY STOPPING
            if(validation_accuracy > prev_val_acc):
                early_stop_counter = 0
                prev_val_acc = validation_accuracy
                print("Early stopping counter: {}".format(early_stop_counter))
                print("Learning rate: {}".format(rate))
                print("Saving model...")
                saver.save(sess, './models/german_recognize-9-1x1/model')
                print()
                continue
            elif(validation_accuracy <= prev_val_acc and early_stop_counter != 25):
                early_stop_counter += 1
                if((rate - rate_decay) < 0):
                    rate_decay *= 0.1
                    rate -= rate_decay
                else:
                    rate -= rate_decay
                print("Early stopping counter: {}".format(early_stop_counter))
                print("Learning rate: {}".format(rate))
                print()
                continue
            elif(validation_accuracy <= prev_val_acc and early_stop_counter == 25):
                print("EARLY STOPPING...")
                print()
                break

        print("Model saved")
        print()

#==============TESTING==============
#===================================
# TEST MODEL ACCURACY
gg = tf.Graph()
with tf.Session(graph = gg) as sess:
    sess.run(tf.global_variables_initializer())
    saver2 = tf.train.import_meta_graph("./models/german_recognize-9-1x1/model.meta")
    saver2.restore(sess, "./models/german_recognize-9-1x1/model")

    prediction = gg.get_tensor_by_name("prediction:0")
    x = gg.get_tensor_by_name("input_data:0")
    keep_prob = gg.get_tensor_by_name("keep_prob:0")


    sess.run(prediction, feed_dict={x: X_test, keep_prob: 1.})

    predictions = (prediction.eval(feed_dict={x: X_test, keep_prob: 1.}))

    conf_mat = sess.run(tf.confusion_matrix(y_test, predictions, n_classes))

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

    with open("german_recognition_results.txt", mode='a') as f:
        f.write("Recognize-9-1x1 Network Results\n")
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
