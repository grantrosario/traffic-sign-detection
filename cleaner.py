import cv2
import numpy as np
from scipy.misc import imread, imsave, imresize
import glob

not_sign_paths = glob.glob('not_sign/*.png')
sign_paths = glob.glob('sign/*/sign_*.png')

for path in sign_paths:
    img = imread(path)
    img = imresize(img, (64, 64))
    imsave(path, img)
