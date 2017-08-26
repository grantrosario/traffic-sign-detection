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

def TestNet(x):
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1

    # subtract mean values from image to normalize
    # mean = tf.constant([118.12, 130.803, 120.883], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
    # x = x-mean

    # Layer 1: Convolutional. Input = 64x64x1. Output = 64x64x8.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(3,3,1,8), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(8))
    conv1_1   = tf.nn.conv2d(x, conv1_W, strides = [1, 1, 1, 1], padding = 'SAME') + conv1_b
    regularizers = tf.nn.l2_loss(conv1_W)
    # Activation.
    conv1_1 = tf.nn.relu(conv1_1)

    # 2) Layer 1-2: Convolutional. Input = 64x64x8. Output = 64x64x8.
    conv1_2_W = tf.Variable(tf.truncated_normal(shape=(3,3,8,8), mean = mu, stddev = sigma))
    conv1_2_b = tf.Variable(tf.zeros(8))
    conv1_2   = tf.nn.conv2d(conv1_1, conv1_2_W, strides = [1, 1, 1, 1], padding = 'SAME') + conv1_2_b
    regularizers = tf.nn.l2_loss(conv1_2_W)
    # Activation.
    conv1_2 = tf.nn.relu(conv1_2)

    # Pooling. Input = 64x64x8. Output = 32x32x8.
    conv1_2 = tf.nn.max_pool(conv1_2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

    # 3) Layer 2-1: Convolutional. Output = 32x32x128.
    conv2_1_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 8, 128), mean = mu, stddev = sigma))
    conv2_1_b = tf.Variable(tf.zeros(128))
    conv2_1   = tf.nn.conv2d(conv1_2, conv2_1_W, strides = [1, 1, 1, 1], padding = 'SAME') + conv2_1_b
    regularizers += tf.nn.l2_loss(conv2_1_W)
    # Activation.
    conv2_1 = tf.nn.relu(conv2_1)

    # 4) Layer 2-2: Convolutional. Output = 32x32x128.
    conv2_2_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 128, 128), mean = mu, stddev = sigma))
    conv2_2_b = tf.Variable(tf.zeros(128))
    conv2_2   = tf.nn.conv2d(conv2_1, conv2_2_W, strides = [1, 1, 1, 1], padding = 'SAME') + conv2_2_b
    regularizers += tf.nn.l2_loss(conv2_2_W)
    # Activation.
    conv2_2 = tf.nn.relu(conv2_2)

    # Pooling. Input = 32x32x128. Output = 16x16x128.
    conv2_2 = tf.nn.max_pool(conv2_2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

    # 5) Layer 3-1: Convolutional. Output = 16x16x256.
    conv3_1_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 128, 256), mean = mu, stddev = sigma))
    conv3_1_b = tf.Variable(tf.zeros(256))
    conv3_1   = tf.nn.conv2d(conv2_2, conv3_1_W, strides = [1, 1, 1, 1], padding = 'SAME') + conv3_1_b
    regularizers += tf.nn.l2_loss(conv3_1_W)
    # Activation.
    conv3_1 = tf.nn.relu(conv3_1)

    # 6) Layer 3-2: Convolutional. Output = 16x16x256.
    conv3_2_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 256, 256), mean = mu, stddev = sigma))
    conv3_2_b = tf.Variable(tf.zeros(256))
    conv3_2   = tf.nn.conv2d(conv3_1, conv3_2_W, strides = [1, 1, 1, 1], padding = 'SAME') + conv3_2_b
    regularizers += tf.nn.l2_loss(conv3_2_W)
    # Activation.
    conv3_2 = tf.nn.relu(conv3_2)

    # 7) Layer 3-3: Convolutional. Output = 16x16x256.
    conv3_3_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 256, 256), mean = mu, stddev = sigma))
    conv3_3_b = tf.Variable(tf.zeros(256))
    conv3_3   = tf.nn.conv2d(conv3_2, conv3_3_W, strides = [1, 1, 1, 1], padding = 'SAME') + conv3_3_b
    regularizers += tf.nn.l2_loss(conv3_3_W)
    # Activation.
    conv3_3 = tf.nn.relu(conv3_3)

    # Pooling. Input = 16x16x256. Output = 8x8x256.
    conv3_3 = tf.nn.max_pool(conv3_3, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

    # 8) Layer 4-1: Convolutional. Output = 8x8x512.
    conv4_1_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 256, 512), mean = mu, stddev = sigma))
    conv4_1_b = tf.Variable(tf.zeros(512))
    conv4_1   = tf.nn.conv2d(conv3_3, conv4_1_W, strides = [1, 1, 1, 1], padding = 'SAME') + conv4_1_b
    regularizers += tf.nn.l2_loss(conv4_1_W)
    # Activation.
    conv4_1 = tf.nn.relu(conv4_1)

    # 9) Layer 4-2: Convolutional. Output = 8x8x512.
    conv4_2_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 512, 512), mean = mu, stddev = sigma))
    conv4_2_b = tf.Variable(tf.zeros(512))
    conv4_2   = tf.nn.conv2d(conv4_1, conv4_2_W, strides = [1, 1, 1, 1], padding = 'SAME') + conv4_2_b
    regularizers += tf.nn.l2_loss(conv4_2_W)
    # Activation.
    conv4_2 = tf.nn.relu(conv4_2)

    # 10) Layer 4-3: Convolutional. Output = 8x8x512.
    conv4_3_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 512, 512), mean = mu, stddev = sigma))
    conv4_3_b = tf.Variable(tf.zeros(512))
    conv4_3   = tf.nn.conv2d(conv4_2, conv4_3_W, strides = [1, 1, 1, 1], padding = 'SAME') + conv4_3_b
    regularizers += tf.nn.l2_loss(conv4_3_W)
    # Activation.
    conv4_3 = tf.nn.relu(conv4_3)

    # Pooling. Input = 8x8x512. Output = 4x4x512.
    conv4_3 = tf.nn.max_pool(conv4_3, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

    # 11) Layer 5-1: Convolutional. Output = 4x4x512.
    conv5_1_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 512, 512), mean = mu, stddev = sigma))
    conv5_1_b = tf.Variable(tf.zeros(512))
    conv5_1   = tf.nn.conv2d(conv4_3, conv5_1_W, strides = [1, 1, 1, 1], padding = 'SAME') + conv5_1_b
    regularizers += tf.nn.l2_loss(conv5_1_W)
    # Activation.
    conv5_1 = tf.nn.relu(conv5_1)

    # 12) Layer 5-2: Convolutional. Output = 4x4x512.
    conv5_2_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 512, 512), mean = mu, stddev = sigma))
    conv5_2_b = tf.Variable(tf.zeros(512))
    conv5_2   = tf.nn.conv2d(conv5_1, conv5_2_W, strides = [1, 1, 1, 1], padding = 'SAME') + conv5_2_b
    regularizers += tf.nn.l2_loss(conv5_2_W)
    # Activation.
    conv5_2 = tf.nn.relu(conv5_2)

    # 13) Layer 5-2: Convolutional. Output = 4x4x512.
    conv5_3_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 512, 512), mean = mu, stddev = sigma))
    conv5_3_b = tf.Variable(tf.zeros(512))
    conv5_3   = tf.nn.conv2d(conv5_2, conv5_3_W, strides = [1, 1, 1, 1], padding = 'SAME') + conv5_3_b
    regularizers += tf.nn.l2_loss(conv5_3_W)
    # Activation.
    conv5_3 = tf.nn.relu(conv5_3)

    # Pooling. Input = 4x4x512. Output = 2x2x512.
    conv5_3 = tf.nn.max_pool(conv5_3, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

    # Flatten. Input = 2x2x512. Output = 2048.
    fc0   = flatten(conv5_3)
    shape = int(np.prod(conv5_3.get_shape()[1:])) #Test to see if this is the same as flatten

    # 14) Layer 6-1: Fully Connected. Input = 2048. Output = 1000.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(shape, 1000), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(1000))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    regularizers += tf.nn.l2_loss(fc1_W)

    # Activation.
    fc1   = tf.nn.relu(fc1)
    fc1   = tf.nn.dropout(fc1, keep_prob)

    # Layer 4: Fully Connected. Input = 800. Output = 500.
    fc2_W = tf.Variable(tf.truncated_normal(shape=(1000,1000), mean = mu, stddev = sigma))
    fc2_b = tf.Variable(tf.zeros(1000))
    fc2   = tf.matmul(fc1, fc2_W) + fc2_b
    regularizers += tf.nn.l2_loss(fc2_W)

    # Activation.
    fc2   = tf.nn.relu(fc2)
    fc2   = tf.nn.dropout(fc2, keep_prob)

    # Layer 5: Fully Connected. Input = 500. Output = 120.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(1000,2), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(2))
    regularizers += tf.nn.l2_loss(fc3_W)
    logits = tf.add(tf.matmul(fc2, fc3_W), fc3_b, "op_logits")

    return [logits, regularizers]
