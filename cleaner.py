import cv2
import numpy as np
from scipy.misc import imread, imsave, imresize
import glob

class DataCleaner():
    """
    """

    def __init__(self, not_sign_dir_path, sign_dir_path):
        self.sign_dir = sign_dir_path
        self.not_sign_dir = not_sign_dir_path

    def clean(self):

        not_sign_paths = glob.glob(self.not_sign_dir + '/*.png')
        sign_paths = glob.glob(self.sign_dir + '/*/*.png')

        self.sizer(not_sign_paths, 64, 64)
        self.sizer(sign_paths, 64, 64)

    def sizer(self, path_list, width, height):

        for path in path_list:
            img = imread(path)
            img = imresize(img, (width, height))
            imsave(path, img)

clean = DataCleaner('0_not_sign', '1_sign')
clean.clean()
