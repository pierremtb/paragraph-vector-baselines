#  prepares data from stanford sentiment tree bank

import re
import numpy as np


class ClassifierDataPrepper:

    def __init__(self, dataPath):
        # read the sentences
        with open(dataPath + "dictionary.txt") as file:
            phrases = file.read().splitlines()
        with open(dataPath + "sentiment_labels.txt") as file:
            classes = file.read().splitlines()
            classes = classes[1:]  # remove first line
        with open(dataPath + "datasetSentences.txt", encoding="utf-8") as file:
            sentences = file.read().splitlines()
            sentences = sentences[1:]  # remove first line
        with open(dataPath + "datasetSplit.txt") as file:
            splitCategories = file.read().splitlines()
            splitCategories = splitCategories[1:]  # remove first line

        self.phrase2index = {}  # returns the phrase index of input sentence string
        # self.index2phrase = {}
        self.phraseClassByPhraseIndex = {}  # returns the sentiment of an input phrase Idx
        self.sentenceIdx2SplitLabel = {}
        self.X = {}
        self.Y = {}

        for sample_class in classes:
            sample_class_split = sample_class.split("|")
            sampleIdx = int(sample_class_split[0])
            score = float(sample_class_split[1])
            self.phraseClassByPhraseIndex[sampleIdx] = score

        for sample_phrase in phrases:
            sample_phrase_split = sample_phrase.split("|")
            sampleIdx = int(sample_phrase_split[1])
            phrase = sample_phrase_split[0]
            # self.index2phrase[sampleIdx] = phrase
            self.phrase2index[phrase] = sampleIdx

        for sentence in sentences:
            sentence_split = re.split(r'\t+', sentence)
            sentenceIdx = int(sentence_split[0])
            sentenceString = sentence_split[1].replace("-LRB-", '(').replace("-RRB-", ')')
            self.X[sentenceIdx] = sentenceString
            self.Y[sentenceIdx] = self.phraseClassByPhraseIndex[self.phrase2index[sentenceString]]

        for splitCategory in splitCategories:
            sentence_split = splitCategory.split(',')
            sentenceIdx = int(sentence_split[0])
            splitLabel = int(sentence_split[1])
            self.sentenceIdx2SplitLabel[sentenceIdx] = splitLabel

    def getXYlabeledBinary(self):
        Y2Binary = {}
        for k, v in self.Y.items():
            if v > 0.5:
                binClass = 1
            else:
                binClass = 0

            Y2Binary[k] = binClass

        x_train = []
        y_train = []
        x_valid = []
        y_valid = []
        x_test = []
        y_test = []

        # 1 = train
        # 2 = test
        # 3 = dev
        for sentIdx, sentence in self.X.items():
            if self.sentenceIdx2SplitLabel[sentIdx] == 1:
                x_train.append(sentence)
                y_train.append(Y2Binary[sentIdx])
            elif self.sentenceIdx2SplitLabel[sentIdx] == 2:
                x_test.append(sentence)
                y_test.append(Y2Binary[sentIdx])
            elif self.sentenceIdx2SplitLabel[sentIdx] == 3:
                x_valid.append(sentence)
                y_valid.append(Y2Binary[sentIdx])
            else:
                print("Error!")

        return x_train, y_train, x_valid, y_valid, x_test, y_test

    # computes the fine-grained labels
    def getXYlabeledSplit(self):
        Y2Split = {}
        for k, v in self.Y.items():
            if v <= 0.2:
                binClass = 0
            elif v <= 0.4:
                binClass = 1
            elif v <= 0.6:
                binClass = 2
            elif v <= 0.8:
                binClass = 3
            else:
                binClass = 4

            Y2Split[k] = binClass

        x_train = []
        y_train = []
        x_valid = []
        y_valid = []
        x_test = []
        y_test = []

        # 1 = train
        # 2 = test
        # 3 = dev
        for sentIdx, sentence in self.X.items():
            if self.sentenceIdx2SplitLabel[sentIdx] == 1:
                x_train.append(sentence)
                y_train.append(Y2Split[sentIdx])
            elif self.sentenceIdx2SplitLabel[sentIdx] == 2:
                x_test.append(sentence)
                y_test.append(Y2Split[sentIdx])
            elif self.sentenceIdx2SplitLabel[sentIdx] == 3:
                x_valid.append(sentence)
                y_valid.append(Y2Split[sentIdx])
            else:
                print("Error!")

        return x_train, y_train, x_valid, y_valid, x_test, y_test

    def cleanhtml(self, raw_html):
        cleanr = re.compile('<.*?>')
        cleantext = cleanr.sub(' ', raw_html)
        cleantext = re.sub(r"([\.\",\(\)!\?;:])", " \\1 ", cleantext)
        re_nonletters = re.compile('[^a-zA-Z ]')
        cleantext = re_nonletters.sub(' ', cleantext)
        return cleantext

