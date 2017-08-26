import numpy as np
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

data_file = "detection_data.p"

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
# fig, axs = plt.subplots(1, 5, figsize=(15, 6))  # create plot boxes for images
# fig.subplots_adjust(hspace = .2, wspace=.1)     # adjust height and width of spacing around boxes
# # axs = axs.ravel(order='C')                      # flatten array into 1-D array
# for i in range(5):
#     index = random.randint(0, len(X_train))
#     image = X_train[index]
#     axs[i].axis('off')
#     axs[i].imshow(image)
#     axs[i].set_title(y_train[index])
# plt.show()

# Plot histogram of training labels used
# plt.figure(figsize=(12, 4))
# hist, bins = np.histogram(y_train, bins = n_classes)
# width = 0.7 * (bins[1] - bins[0]) / 2
# center = (bins[:-1] + bins[1:]) / 2
# barlist = plt.bar(center, hist, align = 'center', width=width, color='royalblue')
# barlist[1].set_color('tomato')
# plt.title("Frequency of labels used")
# plt.xlabel("Label number")
# plt.ylabel("Number of images")
# plt.show()

#========PREPROCESSING==============
#===================================
def preprocess(X):
    t = []
    for i in range(0, len(X)):
        # gray = cv2.cvtColor(X[i], cv2.COLOR_RGB2GRAY)
        # blur = cv2.GaussianBlur(gray, (5,5), 20.0)
        # image = cv2.addWeighted(gray, 2, blur, -1, 0)
        # image = cv2.equalizeHist(image)
        # image = equalize_hist(image)
        image = imresize(X[i], (224, 224))
        t.append(image)
    X = np.reshape(t, (-1, 224, 224, 3))
    print("Image Shape: {}".format(X.shape))
    return X


# show raw image vs processed image
# X_original = X_train
# X_processed = preprocess(X_train)
# fig, axs = plt.subplots(1,2, figsize=(10, 3))
# axs = axs.ravel()
#
# axs[0].axis('off')
# axs[0].set_title('Original')
# axs[0].imshow(X_original[156].squeeze())
#
# axs[1].axis('off')
# axs[1].set_title('Processed')
# axs[1].imshow(X_processed[156].squeeze(), cmap='gray')

print("Training data")
X_train = preprocess(X_train)
print("Validation data")
X_valid = preprocess(X_valid)
print("Test data")
X_test = preprocess(X_test)
X_train, y_train = shuffle(X_train, y_train)

#=========BUILD ARCHITECTURE========
#===================================
### Define architecture.
EPOCHS = 10
BATCH_SIZE = 64
beta = 0.001

def LeNet(x):
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    # subtract mean values from image to normalize
    mean = tf.constant([118.12, 130.803, 120.883], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
    x = x-mean

    # 1) Layer 1-1: Convolutional. Input = 224x224x3. Output = 224x224x64.
    conv1_1_W = tf.Variable(tf.truncated_normal(shape=(3,3,3,64), mean = mu, stddev = sigma))
    conv1_1_b = tf.Variable(tf.zeros(64))
    conv1_1   = tf.nn.conv2d(x, conv1_1_W, strides = [1, 1, 1, 1], padding = 'SAME') + conv1_1_b
    regularizers = tf.nn.l2_loss(conv1_1_W)
    # Activation.
    conv1_1 = tf.nn.relu(conv1_1)

    # 2) Layer 1-2: Convolutional. Input = 224x224x64. Output = 224x224x64.
    conv1_2_W = tf.Variable(tf.truncated_normal(shape=(3,3,64,64), mean = mu, stddev = sigma))
    conv1_2_b = tf.Variable(tf.zeros(64))
    conv1_2   = tf.nn.conv2d(conv1_1, conv1_2_W, strides = [1, 1, 1, 1], padding = 'SAME') + conv1_2_b
    regularizers = tf.nn.l2_loss(conv1_2_W)
    # Activation.
    conv1_2 = tf.nn.relu(conv1_2)

    # Pooling. Input = 224x224x64. Output = 112x112x64.
    conv1_2 = tf.nn.max_pool(conv1_2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

    # 3) Layer 2-1: Convolutional. Output = 112x112x128.
    conv2_1_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 64, 128), mean = mu, stddev = sigma))
    conv2_1_b = tf.Variable(tf.zeros(128))
    conv2_1   = tf.nn.conv2d(conv1_2, conv2_1_W, strides = [1, 1, 1, 1], padding = 'SAME') + conv2_1_b
    regularizers += tf.nn.l2_loss(conv2_1_W)
    # Activation.
    conv2_1 = tf.nn.relu(conv2_1)

    # 4) Layer 2-2: Convolutional. Output = 112x112x128.
    conv2_2_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 128, 128), mean = mu, stddev = sigma))
    conv2_2_b = tf.Variable(tf.zeros(128))
    conv2_2   = tf.nn.conv2d(conv2_1, conv2_2_W, strides = [1, 1, 1, 1], padding = 'SAME') + conv2_2_b
    regularizers += tf.nn.l2_loss(conv2_2_W)
    # Activation.
    conv2_2 = tf.nn.relu(conv2_2)

    # Pooling. Input = 112x112x128. Output = 56x56x128.
    conv2_2 = tf.nn.max_pool(conv2_2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

    # 5) Layer 3-1: Convolutional. Output = 56x56x256.
    conv3_1_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 128, 256), mean = mu, stddev = sigma))
    conv3_1_b = tf.Variable(tf.zeros(256))
    conv3_1   = tf.nn.conv2d(conv2_2, conv3_1_W, strides = [1, 1, 1, 1], padding = 'SAME') + conv3_1_b
    regularizers += tf.nn.l2_loss(conv3_1_W)
    # Activation.
    conv3_1 = tf.nn.relu(conv3_1)

    # 6) Layer 3-2: Convolutional. Output = 56x56x256.
    conv3_2_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 256, 256), mean = mu, stddev = sigma))
    conv3_2_b = tf.Variable(tf.zeros(256))
    conv3_2   = tf.nn.conv2d(conv3_1, conv3_2_W, strides = [1, 1, 1, 1], padding = 'SAME') + conv3_2_b
    regularizers += tf.nn.l2_loss(conv3_2_W)
    # Activation.
    conv3_2 = tf.nn.relu(conv3_2)

    # 7) Layer 3-3: Convolutional. Output = 56x56x256.
    conv3_3_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 256, 256), mean = mu, stddev = sigma))
    conv3_3_b = tf.Variable(tf.zeros(256))
    conv3_3   = tf.nn.conv2d(conv3_2, conv3_3_W, strides = [1, 1, 1, 1], padding = 'SAME') + conv3_3_b
    regularizers += tf.nn.l2_loss(conv3_3_W)
    # Activation.
    conv3_3 = tf.nn.relu(conv3_3)

    # Pooling. Input = 56x56x256. Output = 28x28x256.
    conv3_3 = tf.nn.max_pool(conv3_3, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

    # 8) Layer 4-1: Convolutional. Output = 28x28x512.
    conv4_1_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 256, 512), mean = mu, stddev = sigma))
    conv4_1_b = tf.Variable(tf.zeros(512))
    conv4_1   = tf.nn.conv2d(conv3_3, conv4_1_W, strides = [1, 1, 1, 1], padding = 'SAME') + conv4_1_b
    regularizers += tf.nn.l2_loss(conv4_1_W)
    # Activation.
    conv4_1 = tf.nn.relu(conv4_1)

    # 9) Layer 4-2: Convolutional. Output = 28x28x512.
    conv4_2_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 512, 512), mean = mu, stddev = sigma))
    conv4_2_b = tf.Variable(tf.zeros(512))
    conv4_2   = tf.nn.conv2d(conv4_1, conv4_2_W, strides = [1, 1, 1, 1], padding = 'SAME') + conv4_2_b
    regularizers += tf.nn.l2_loss(conv4_2_W)
    # Activation.
    conv4_2 = tf.nn.relu(conv4_2)

    # 10) Layer 4-3: Convolutional. Output = 28x28x512.
    conv4_3_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 512, 512), mean = mu, stddev = sigma))
    conv4_3_b = tf.Variable(tf.zeros(512))
    conv4_3   = tf.nn.conv2d(conv4_2, conv4_3_W, strides = [1, 1, 1, 1], padding = 'SAME') + conv4_3_b
    regularizers += tf.nn.l2_loss(conv4_3_W)
    # Activation.
    conv4_3 = tf.nn.relu(conv4_3)

    # Pooling. Input = 28x28x512. Output = 14x14x512.
    conv4_3 = tf.nn.max_pool(conv4_3, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

    # 11) Layer 5-1: Convolutional. Output = 14x14x512.
    conv5_1_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 512, 512), mean = mu, stddev = sigma))
    conv5_1_b = tf.Variable(tf.zeros(512))
    conv5_1   = tf.nn.conv2d(conv4_3, conv5_1_W, strides = [1, 1, 1, 1], padding = 'SAME') + conv5_1_b
    regularizers += tf.nn.l2_loss(conv5_1_W)
    # Activation.
    conv5_1 = tf.nn.relu(conv5_1)

    # 12) Layer 5-2: Convolutional. Output = 14x14x512.
    conv5_2_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 512, 512), mean = mu, stddev = sigma))
    conv5_2_b = tf.Variable(tf.zeros(512))
    conv5_2   = tf.nn.conv2d(conv5_1, conv5_2_W, strides = [1, 1, 1, 1], padding = 'SAME') + conv5_2_b
    regularizers += tf.nn.l2_loss(conv5_2_W)
    # Activation.
    conv5_2 = tf.nn.relu(conv5_2)

    # 13) Layer 5-2: Convolutional. Output = 14x14x512.
    conv5_3_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 512, 512), mean = mu, stddev = sigma))
    conv5_3_b = tf.Variable(tf.zeros(512))
    conv5_3   = tf.nn.conv2d(conv5_2, conv5_3_W, strides = [1, 1, 1, 1], padding = 'SAME') + conv5_3_b
    regularizers += tf.nn.l2_loss(conv5_3_W)
    # Activation.
    conv5_3 = tf.nn.relu(conv5_3)

    # Pooling. Input = 14x14x512. Output = 7x7x512.
    conv5_3 = tf.nn.max_pool(conv5_3, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

    # Flatten. Input = 7x7x512. Output = 2704.
    fc0   = flatten(conv5_3)
    shape = int(np.prod(conv5_3.get_shape()[1:])) #Test to see if this is the same as flatten

    # 14) Layer 6-1: Fully Connected. Input = 2704. Output = 800.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(shape, 4096), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(4096))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    regularizers += tf.nn.l2_loss(fc1_W)

    # Activation.
    fc1   = tf.nn.relu(fc1)
    fc1   = tf.nn.dropout(fc1, keep_prob)

    # Layer 4: Fully Connected. Input = 800. Output = 500.
    fc2_W = tf.Variable(tf.truncated_normal(shape=(4096,4096), mean = mu, stddev = sigma))
    fc2_b = tf.Variable(tf.zeros(4096))
    fc2   = tf.matmul(fc1, fc2_W) + fc2_b
    regularizers += tf.nn.l2_loss(fc2_W)

    # Activation.
    fc2   = tf.nn.relu(fc2)
    fc2   = tf.nn.dropout(fc2, keep_prob)

    # Layer 5: Fully Connected. Input = 500. Output = 120.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(4096,2), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(2))
    regularizers += tf.nn.l2_loss(fc3_W)
    logits = tf.add(tf.matmul(fc2, fc3_W), fc3_b, "op_logits")

    return [logits, regularizers]


rate = 0.0008
keep_prob = tf.placeholder(tf.float32, name="keep_prob") # probablity of keeping for dropout
x = tf.placeholder(tf.float32, (None, 224, 224, 3), name="input_data")
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 2)

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

        print()
        print("Training...")
        for i in range(EPOCHS):
            print("EPOCH {} ...".format(i+1))
            X_train, y_train = shuffle(X_train, y_train)
            for offset in tqdm(range(0, num_examples, BATCH_SIZE)):
                end = offset + BATCH_SIZE
                batch_x, batch_y = X_train[offset:end], y_train[offset:end]
                sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})

            print()
            print("Evaluating validation accuracy...")
            validation_accuracy = evaluate(X_valid, y_valid)
            print("Evaluating training accuracy...")
            training_accuracy = evaluate(X_train, y_train)
            # test_accuracy = evaluate(X_test, y_test)
            print("Validation Accuracy = {:.3f}".format(validation_accuracy))
            print("Training Accuracy = {:.3f}".format(training_accuracy))
            # print("Test Accuracy = {:.3f}".format(test_accuracy))
            print()

        saver.save(sess, './test_net/model')
        print("Model saved")

#==============TESTING==============
#===================================
# TEST MODEL ACCURACY
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver2 = tf.train.import_meta_graph('./test_net/model.meta')
    saver2.restore(sess, "./test_net/model")
    test_accuracy = evaluate(X_test, y_test)
    print("Test Set Accuracy = {:.3f}".format(test_accuracy))
