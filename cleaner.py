import cv2
import numpy as np
import os
import glob
import pickle
from scipy.misc import imread, imsave, imresize
from sklearn.model_selection import train_test_split as trainTestSplit
from sklearn.utils import shuffle


class DataCleaner():
    """
    """

    def __init__(self, notSignDirPath, signDirPath):
        self.signDir = signDirPath
        self.notSignDir = notSignDirPath

    def sizer(self, pathList, width, height):

        for path in pathList:
            img = imread(path)
            img = imresize(img, (width, height))
            imsave(path, img)

    def cleanData(self):

        notSignPaths = glob.glob(self.notSignDir + '/*.png')
        signPaths = glob.glob(self.signDir + '/*/*.png')

        self.sizer(notSignPaths, 64, 64)
        self.sizer(signPaths, 64, 64)

    def getData(self):

        dataFile = 'dataset.p'

        if not os.path.isfile(dataFile):
            print("\nAttempting to generate new data file...\n")

            signFolder = '1_sign/'
            nonSignFolder = '0_not_sign/'

            if not os.path.isdir(signFolder) or not os.path.isdir(nonSignFolder):
                print("No samples found.\nExiting.")
                return None, None, None, None, None, None
            else:
                signFiles = glob.glob('{}*/*.png'.format(signFolder))
                nonSignFiles = glob.glob('{}*.png'.format(nonSignFolder))

                imageSamplesFiles = signFiles + nonSignFiles
                y = np.concatenate((np.ones(len(signFiles)), np.zeros(len(nonSignFiles))))

                imageSamplesFiles, y = shuffle(imageSamplesFiles, y)

                xTrain, xTest, yTrain, yTest = trainTestSplit(imageSamplesFiles, y, test_size=0.2, random_state=42)

                xTrain, xVal, yTrain, yVal = trainTestSplit(xTrain, yTrain, test_size=0.2, random_state=42)

                data = {'xTrain': xTrain, 'xValidation': xVal, 'xTest': xTest,
                        'yTrain': yTrain, 'yValidation': yVal, 'yTest': yTest}

                pickle.dump(data, open(dataFile, 'wb'))

                return xTrain, xVal, xTest, yTrain, yVal, yTest
        else:
            with open(dataFile, mode='rb') as f:
                data = pickle.load(f)

                xTrain = data['xTrain']
                xValidation = data['xValidation']
                xTest = data['xTest']
                yTrain = data['yTrain']
                yValidation = data['yValidation']
                yTest = data['yTest']

                return xTrain, xValidation, xTest, yTrain, yValidation, yTest




clean = DataCleaner('0_not_sign', '1_sign')
clean.cleanData()
xTrain, xVal, xTest, yTrain, yVal, yTest = clean.getData()
print(len(xTrain))
print(len(xVal))
print(len(xTest))
print(len(yTrain))
print(len(yVal))
print(len(yTest))
