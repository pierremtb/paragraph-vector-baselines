from ClassifierDataPrepper import ClassifierDataPrepper
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn import metrics

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

print("Loading models ...")
models = loadModelsFromPickles(isFineGrainedMode)

print("Making predictions using loaded models ...")
predictions = []
for model in models:
    prediction = model.predict(X_train_count)
    predictions.append(prediction)

# transposing list of lists
predictionsFlipped = list(map(list, zip(*predictions)))

# create and train Logistic Regression model for each type of vectorizer
print("Training meta classifier...")
metaModel = LogisticRegression()
metaModel.fit(predictionsFlipped, y_train)

# run models on validation set
print("Running models...")
print("Making predictions using loaded models ...")
predictions = []
for model in models:
    prediction = model.predict(X_validate_count)
    predictions.append(prediction)

# transposing list of lists
predictionsFlipped = list(map(list, zip(*predictions)))

meta_predictions = metaModel.predict(predictionsFlipped)

# print model performance
meta_accuracy = metrics.accuracy_score(y_valid, meta_predictions)

print("Meta Model Accuracy:")
print(meta_accuracy)
