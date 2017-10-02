import cv2
import numpy as np
import os
import glob
import pickle
from scipy.misc import imread, imsave, imresize
from sklearn.model_selection import train_test_split as trainTestSplit
from sklearn.utils import shuffle


class DataMaker:

    def __init__(self, notSignDirPath='0_not_sign/', signDirPath='1_sign/'):
        self.notSignFolder = notSignDirPath
        self.notSignFiles = glob.glob('{}*.png'.format(self.notSignFolder))
        self.signFolder = signDirPath
        self.signFiles = glob.glob('{}*/*.png'.format(self.signFolder))
        self.testSigns = glob.glob('test_data/signs/*/*.png')


    def sizer(self, pathList, width, height):
        """
        Returns the image data resized to (64, 64, 3)
        :param pathList: List of image paths
        :param width: Desired width of images
        :param height: Desired height of images
        """
        for path in pathList:
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = imresize(img, (width, height))
            imsave(path, img)


    def processData(self):
        """
        Asks user if images need to be resized in case there is new data.
        If yes, calls sizer(), if no, then nothing happens.
        """
        files = []
        for f in self.signFiles:
            files.append(f)
        for f in self.notSignFiles:
            files.append(f)
        for f in self.testSigns:
            files.append(f)
        if((input('Do images need resized? (y/n): ')) == 'y'):
            self.sizer(files, 64, 64)


    def getDetectionData(self):
        """
        Gather the images and labels for detecting signs.
        Returns a dictionary with Training, Validation, and Test data and labels
        Also creates a pickle file for storing dictionary
        """
        dataDetectionFile = 'detection_data.p'

        if not os.path.isfile(dataDetectionFile):
            print("\nAttempting to generate new detection data file...\n")

            if not os.path.isdir(self.signFolder) or not os.path.isdir(self.notSignFolder):
                print("No samples found.\nExiting.")
                return None

            else:
                imageSampleFiles = self.signFiles + self.notSignFiles
                pathTrack = []
                for path in imageSampleFiles:
                    if (path == imageSampleFiles[0]):
                        pathTrack.append(path)
                        img = cv2.imread(path)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        signFeatures = np.array([img])

                    else:
                        pathTrack.append(path)
                        img = cv2.imread(path)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        newArray = np.array([img])
                        signFeatures = np.concatenate([signFeatures, newArray])


                y = np.concatenate((np.ones(len(self.signFiles)), np.zeros(len(self.notSignFiles))))

                signFeatures, y = shuffle(signFeatures, y)


                xTrain, xTest, yTrain, yTest = trainTestSplit(signFeatures, y, test_size=0.2, random_state=42)

                xTrain, xVal, yTrain, yVal = trainTestSplit(xTrain, yTrain, test_size=0.2, random_state=42)

                detectionData = {'xTrain': xTrain,
                                 'xValidation': xVal,
                                 'xTest': xTest,
                                 'yTrain': yTrain,
                                 'yValidation': yVal,
                                 'yTest': yTest}

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
        """
        Gather the images and labels for recognizing signs.
        Returns a dictionary with Training, Validation, and Test data
        Also creates a pickle file for storing dictionary
        """

        dataRecognitionFile = 'recognition_data.p'

        if not os.path.isfile(dataRecognitionFile):
            print("\nAttempting to generate new recognition data file...\n")

            signFolder = '1_sign/'
            testSignFolder = 'test_data/signs/'

            if not os.path.isdir(signFolder):
                print("No samples found.\nExiting.")
                return None
            else:
                signLabels = []
                uniqueSignFolders = glob.glob('{}*/'.format(signFolder))
                uniqueSignPaths = glob.glob('{}*/*.png'.format(signFolder))

                testLabels = []
                uniqueTestFolders = glob.glob('{}*/'.format(testSignFolder))
                uniqueTestPaths = glob.glob('{}*/*.png'.format(testSignFolder))

                for path in uniqueSignPaths:
                    if (path == uniqueSignPaths[0]):
                        imageLabel = int((path.split("/")[-1]).split("_")[0])
                        signLabels.append(imageLabel)
                        img = cv2.imread(path)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        signFeatures = np.array([img])
                    else:
                        imageLabel = int((path.split("/")[-1]).split("_")[0])
                        signLabels.append(imageLabel)
                        img = cv2.imread(path)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        newArray = np.array([img])
                        signFeatures = np.concatenate([signFeatures, newArray])

                for path in uniqueTestPaths:
                    if (path == uniqueTestPaths[0]):
                        testLabel = int((path.split("/")[-1]).split("_")[0])
                        testLabels.append(testLabel)
                        img = cv2.imread(path)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        testFeatures = np.array([img])
                    else:
                        testLabel = int((path.split("/")[-1]).split("_")[0])
                        testLabels.append(testLabel)
                        img = cv2.imread(path)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        newArray = np.array([img])
                        testFeatures = np.concatenate([testFeatures, newArray])


                signFeatures, signLabels = shuffle(signFeatures, signLabels)
                testFeatures, testLabels = shuffle(testFeatures, testLabels)

                xTrain, yTrain = signFeatures, signLabels
                xTest, yTest = testFeatures, testLabels

                xTrain, xVal, yTrain, yVal = trainTestSplit(xTrain, yTrain, test_size=0.2, random_state=42)


                recognitionData = {'xTrain': xTrain, 'xValidation': xVal, 'xTest': xTest,
                                   'yTrain': yTrain, 'yValidation': yVal, 'yTest': yTest}

                pickle.dump(recognitionData, open(dataRecognitionFile, 'wb'))

                return recognitionData
        else:
            with open(dataRecognitionFile, mode='rb') as f:
                data = pickle.load(f)
                return data

def main():
    """
    Pipeline to run if called from command line
    """
    clean = DataMaker('0_not_sign/', '1_sign/')
    clean.processData()
    detectionData = clean.getDetectionData()
    with open("detection_results.txt", mode='w') as f:
        f.write("--------------------\n")
        f.write("Data Created\n")
        f.write("---\n")
        f.write("{} training images\n".format(len(detectionData['xTrain'])))
        f.write("{} validation images\n".format(len(detectionData['xValidation'])))
        f.write("{} test images\n".format(len(detectionData['xTest'])))
        f.write("--------------------\n")

    recognitionData = clean.getRecognitionData()
    with open("recognition_results.txt", mode='w') as f:
        f.write("--------------------\n")
        f.write("Data Created\n")
        f.write("---\n")
        f.write("{} training images\n".format(len(recognitionData['xTrain'])))
        f.write("{} validation images\n".format(len(recognitionData['xValidation'])))
        f.write("{} test images\n".format(len(recognitionData['xTest'])))
        f.write("--------------------\n")

    with open("german_recognition_results.txt", mode='w') as f:
        f.write("--------------------\n")
        f.write("Data Created\n")
        f.write("--------------------\n")
        f.write("--------------------\n")


if __name__ == '__main__':
    main()
