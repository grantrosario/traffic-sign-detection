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

    def getMeans(pathList, dataType):
        r_values = []
        g_values = []
        b_values = []
        count = 0
        for path in pathList:
            count += 1
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            for r_array in img[:,:,0]:
                for r_val in r_array:
                    r_values.append(r_val)
            for g_array in img[:,:,1]:
                for g_val in g_array:
                    g_values.append(g_val)
            for b_array in img[:,:,2]:
                for b_val in b_array:
                    b_values.append(b_val)
        if(dataType == "detection"):
            print("{} detection images: mean is [r,g,b] => [{}, {}, {}]".format(count,
                                                                               (sum(r_values)/len(r_values)),
                                                                               (sum(g_values)/len(g_values)),
                                                                               (sum(b_values)/len(b_values))))
        if(dataType == "recognition"):
            print("{} recognition images: mean is [r,g,b] => [{}, {}, {}]".format(count,
                                                                                 (sum(r_values)/len(r_values)),
                                                                                 (sum(g_values)/len(g_values)),
                                                                                 (sum(b_values)/len(b_values))))



    def processData(self):
        """
        Asks user if images need to be resized in case there is new data.
        If yes, calls sizer(), if no, then nothing happens.
        """
        files = []
        if((input('Do images need resized? (y/n): ')) == 'y'):
            for f in self.signFiles:
                files.append(f)
            for f in self.notSignFiles:
                files.append(f)
            self.sizer(files, 64, 64)
        if((input('Calculate new means? (y/n): ')) == 'y'):
            dataType = input('Detection or Recognition? (d/r): ')
            if(dataType == 'd'):
                self.getMeans(files, "detection")
            elif(dataType == 'r'):
                self.getMeans(files, "recognition")


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

            if not os.path.isdir(signFolder):
                print("No samples found.\nExiting.")
                return None
            else:
                signLabels = []
                uniqueSignFolders = glob.glob('{}*/'.format(signFolder))
                uniqueSignPaths = glob.glob('{}*/*.png'.format(signFolder))

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

                signFeatures, signLabels = shuffle(signFeatures, signLabels)


                xTrain, xTest, yTrain, yTest = trainTestSplit(signFeatures, signLabels, test_size=0.2, random_state=42)

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
    #recognitionData = clean.getRecognitionData()


if __name__ == '__main__':
    main()
