#  prepares data from stanford sentiment tree bank

import re
import numpy as np


class ClassifierDataPrepper:

    def __init__(self, dataPath):
        # read the sentences
        with open(dataPath + "dictionary.txt", encoding='ISO-8859-1') as file:
            phrases = file.read().splitlines()
        with open(dataPath + "sentiment_labels.txt", encoding='ISO-8859-1') as file:
            classes = file.read().splitlines()
            classes = classes[1:]  # remove first line

        self.X = {}
        self.Y = {}

        for sample_class in classes:
            sample_class_split = sample_class.split("|")
            sampleIdx = int(sample_class_split[0])
            score = float(sample_class_split[1])
            self.Y[sampleIdx] = score

        for sample_phrase in phrases:
            sample_phrase_split = sample_phrase.split("|")
            sampleIdx = int(sample_phrase_split[1])
            phrase = sample_phrase_split[0]
            self.X[sampleIdx] = phrase

    def getXYlabeledBinary(self):
        Y2Binary = {}
        for k, v in self.Y.items():
            if v > 0.5:
                binClass = 1
            else:
                binClass = 0

            Y2Binary[k] = binClass

        return self.X, Y2Binary

    def cleanhtml(self, raw_html):
        cleanr = re.compile('<.*?>')
        cleantext = cleanr.sub(' ', raw_html)
        cleantext = re.sub(r"([\.\",\(\)!\?;:])", " \\1 ", cleantext)
        re_nonletters = re.compile('[^a-zA-Z ]')
        cleantext = re_nonletters.sub(' ', cleantext)
        return cleantext

