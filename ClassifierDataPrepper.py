# Class to quickly parse data from folders into pandas dataframes
# COMP 551
# Boury Mbodj
# Humayun Khan
# Michael Segev
# Feb 14 2019
import DataGrabber
import pandas as pd
import re

class ClassifierDataPrepper:

    # grabs all files from 3 provided folders and reads them into lists
    def __init__(self, positiveTrainingDataPath, negativeTrainingDataPath, testDataPath):
        self.dataGrabberPosTrain = None
        self.dataGrabberNegTrain = None
        self.dataGrabberTest = None

        if positiveTrainingDataPath is not None:
            self.dataGrabberPosTrain = DataGrabber.DataGrabber(positiveTrainingDataPath)
        if negativeTrainingDataPath is not None:
            self.dataGrabberNegTrain = DataGrabber.DataGrabber(negativeTrainingDataPath)

        if testDataPath is not None:
            self.dataGrabberTest = DataGrabber.DataGrabber(testDataPath)

        self.trainDf = None
        self.testDf = None

        self.getDataFrames()

    # converts review lists into data frames
    def getDataFrames(self):
        if self.dataGrabberPosTrain is not None and self.dataGrabberNegTrain is not None:
            self.trainDf = pd.concat([
                pd.DataFrame({"review": self.dataGrabberPosTrain.readCommentFiles(), "label": 1}),
                pd.DataFrame({"review": self.dataGrabberNegTrain.readCommentFiles(), "label": 0}),
            ],)
        if self.dataGrabberTest is not None:
            self.testDf = pd.DataFrame({"review": self.dataGrabberTest.readCommentFiles(), "label": -1})

    # extracts only the training data from dataframes
    def getXYlabeled(self):
        Xtrain = self.trainDf.review
        Ytrain = self.trainDf.label
        return Xtrain, Ytrain

    #
    def getXtest(self):
        Xtest = self.testDf.review
        return Xtest

    def cleanhtml(self, raw_html):
        cleanr = re.compile('<.*?>')
        cleantext = cleanr.sub(' ', raw_html)
        cleantext = re.sub(r"([\.\",\(\)!\?;:])", " \\1 ", cleantext)
        re_nonletters = re.compile('[^a-zA-Z ]')
        cleantext = re_nonletters.sub(' ', cleantext)
        return cleantext

