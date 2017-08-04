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
        """
        Returns the image data resized to (64, 64, 3) only if the first image
        is not already (64, 64, 3), otherwise, they are already the proper shape
        so this will pass
        :param pathList: List of image paths
        :param width: Desired width of images
        :param height: Desired height of images
        """
        for path in pathList:
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = imresize(img, (width, height))
            imsave(path, img)


    def cleanData(self):

        notSignPaths = glob.glob(self.notSignDir + '/*.png')
        signPaths = glob.glob(self.signDir + '/*/*.png')

        if((input('Do images need resized? (y/n): ')) == 'y'):
            self.sizer(notSignPaths, 64, 64)
            self.sizer(signPaths, 64, 64)

    def getDetectionData(self):

        dataDetectionFile = 'data_detection.p'

        if not os.path.isfile(dataDetectionFile):
            print("\nAttempting to generate new detection data file...\n")

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

                detectionData = {'xTrain': xTrain, 'xValidation': xVal, 'xTest': xTest,
                        'yTrain': yTrain, 'yValidation': yVal, 'yTest': yTest}

                pickle.dump(detectionData, open(dataDetectionFile, 'wb'))

                return detectionData
        else:
            with open(dataDetectionFile, mode='rb') as f:
                data = pickle.load(f)

                detectionData = {'xTrain': data['xTrain'],
                                 'xValidation': data['xValidation'],
                                 'xTest': data['xTest'],
                                 'yTrain': data['yTrain'],
                                 'yValidation': data['yValidation'],
                                 'yTest': data['yTest']}

                return detectionData

    def getRecognitionData(self):

        dataRecognitionFile = 'data_recognition.p'

        if not os.path.isfile(dataRecognitionFile):
            print("\nAttempting to generate new recognition data file...\n")

            signFolder = '1_sign/'

            if not os.path.isdir(signFolder):
                print("No samples found.\nExiting.")
                return None, None, None, None, None, None
            else:
                signLabels = []
                uniqueSignFolders = glob.glob('{}*/'.format(signFolder))
                uniqueSignPaths = glob.glob('{}*/*.png'.format(signFolder))

                for path in uniqueSignFolders:
                    signLabels.append(int(path.split('/')[1].split('_')[0]))
                signLabels = (sorted(signLabels))

                img = cv2.imread(uniqueSignPaths[0])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                print(img)
                return

                imageSamplesFiles = signFiles + nonSignFiles
                y = np.concatenate((np.ones(len(signFiles)), np.zeros(len(nonSignFiles))))

                imageSamplesFiles, y = shuffle(imageSamplesFiles, y)

                xTrain, xTest, yTrain, yTest = trainTestSplit(imageSamplesFiles, y, test_size=0.2, random_state=42)

                xTrain, xVal, yTrain, yVal = trainTestSplit(xTrain, yTrain, test_size=0.2, random_state=42)

                data = {'xTrain': xTrain, 'xValidation': xVal, 'xTest': xTest,
                        'yTrain': yTrain, 'yValidation': yVal, 'yTest': yTest}

                pickle.dump(data, open(dataDetectionFile, 'wb'))

                return xTrain, xVal, xTest, yTrain, yVal, yTest
        else:
            with open(dataDetectionFile, mode='rb') as f:
                data = pickle.load(f)

                xTrain = data['xTrain']
                xValidation = data['xValidation']
                xTest = data['xTest']
                yTrain = data['yTrain']
                yValidation = data['yValidation']
                yTest = data['yTest']

                return xTrain, xValidation, xTest, yTrain, yValidation, yTest


clean = DataCleaner('0_not_sign', '1_sign')
#clean.cleanData()
#detectionData = clean.getDetectionData()
#clean.getRecognitionData()
