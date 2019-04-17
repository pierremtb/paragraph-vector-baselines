# COMP 551 Mini Project 4
# 2019-04-17
# Segev, Michael
# Jacquier, Pierre
# Han, Zhenze
# Helper functions to save and load models to file once they are trained

import pickle
from os import listdir
from os.path import isfile, join


def saveBestModel(models, accuracies, isFineGrainedMode, modelTypeString):
    saveBestPickle = int(input("Save best model to pickle file?\n\t0 -> No\n\t1 -> Yes\n"))

    if saveBestPickle == 1:
        bestAccuracy = 0
        bestModelIdx = 0
        for modelIdx, accuracy in enumerate(accuracies):
            if accuracy > bestAccuracy:
                bestAccuracy = accuracy
                bestModelIdx = modelIdx
        if isFineGrainedMode:
            modeString = "fine_grain"
        else:
            modeString = "binary"

        filename = "models/{}/{}_model.pckl".format(modeString, modelTypeString)
        pickle.dump(models[bestModelIdx], open(filename, 'wb'))


def saveModel(model, isFineGrainedMode, modelTypeString):
    savePickle = int(input("Save model to pickle file?\n\t0 -> No\n\t1 -> Yes\n"))

    if savePickle == 1:
        if isFineGrainedMode:
            modeString = "fine_grain"
        else:
            modeString = "binary"

        filename = "models/{}/{}_model.pckl".format(modeString, modelTypeString)
        pickle.dump(model, open(filename, 'wb'))


def loadModelsFromPickles(isFineGrainedMode):
    path = "models/"
    if isFineGrainedMode == 1:
        path += "fine_grain/"
    else:
        path += "binary/"

    filenamess = [f for f in listdir(path) if isfile(join(path, f))]

    models = []
    for filename in filenamess:
        print("Loading {}".format(path+filename))
        loaded_model = pickle.load(open(path+filename, 'rb'))
        models.append(loaded_model)
    return models
