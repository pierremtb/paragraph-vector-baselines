from ClassifierDataPrepper import ClassifierDataPrepper
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier

from helpers import *

dataPath = "./stanfordSentimentTreebank/"

print("Opening files...")
cdp = ClassifierDataPrepper(dataPath)

isFineGrainedMode = int(input("Binary or Fine Grained?\n\t0 -> Binary\n\t1 -> Fine Grained\n"))

if isFineGrainedMode == 0:
    print("Binary!")
    x_train, y_train, x_valid, y_valid, x_test, y_test = cdp.getXYlabeledBinary()
else:
    print("Fine Grained!")
    x_train, y_train, x_valid, y_valid, x_test, y_test = cdp.getXYlabeledSplit()


# uncomment this to use test set!
x_valid = x_test
y_valid = y_test

# Building different vectorizers used to parse the text into features
print("Extracting features from data ...")
vectCount = CountVectorizer(ngram_range=(1, 2), binary=False)

# learn the vocabularies from training data for each vector type
vectCount.fit(x_train)

# transform training data
X_train_count = vectCount.transform(x_train)

# transform validation data
X_validate_count = vectCount.transform(x_valid)

print("Training model...")
models = []
model = ExtraTreesClassifier(n_estimators=500, max_depth=None, min_samples_split=2, random_state=0)
models.append(model)
model.fit(X_train_count, y_train)

# run model on validation set
print("Running models...")
predictions = model.predict(X_validate_count)

# print model performance
accuracies = []
accuracy = metrics.accuracy_score(y_valid, predictions)
accuracies.append(accuracy)

print("Performance:")
print(accuracy)

# saveBestModel(models, accuracies, isFineGrainedMode, "SVM")
saveModel(model, isFineGrainedMode, "DT")

