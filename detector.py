import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import random
import time
from collections import deque
from skimage.feature import hog
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from scipy.ndimage.measurements import label

class detector():

    def __init__(self, image):
        self.testImage = image

    def detect(self.testImage, ystart, ystop, scale, nn, xScaler, windowSize):
        pass
