import tensorflow as tf
import numpy
import matplotlib.pyplot as plt

class NN():

    def __init__(self):

        self.model
        self.k_output
        self.image_width
        self.image_height
        self.color_channels

        self.filter_size_width
        self.filter_size_height

    def convnet():

        # Input/Image
        input_ = tf.placeholder(
                tf.float32,
                shape = [None,
                         self.image_height,
                         self.image_width,
                         self.color_channels])

        # Weight and bias
        weight = tf.Variable(tf.truncated_normal(
                [self.filter_size_height,
                 self.filter_size_width,
                 self.color_channels,
                 self.k_output]))

        bias = tf.Variable(tf.zeros(k_output))

        # Apply Convolution
        conv_layer = tf.nn.conv2d(input_, weight, strides=[1, 2, 2, 1], padding='SAME')
        # Add bias
        conv_layer = tf.nn.bias_add(conv_layer, bias)
        # Apply activation function
        conv_layer = tf.nn.relu(conv_layer)
