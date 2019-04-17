# COMP 551 Mini Project 4
# 2019-04-17
# Segev, Michael
# Jacquier, Pierre
# Han, Zhenze
# Run this file to test different feature extraction pipelines with a NB classifier on Stanford Sentiment Treebank

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
# Split labelled data into train and validation sets
# X_train, X_validate, Y_train, Y_validate = train_test_split(Xl, Y1, test_size=0.2, random_state=99)

# Building different vectorizers used to parse the text into features
print("Extracting features from data ...")
vectBinCount = CountVectorizer(ngram_range=(1, 2), binary=True)
vectCount = CountVectorizer(ngram_range=(1, 2), binary=False)
vectTfidf = TfidfVectorizer(ngram_range=(1, 2))

# learn the vocabularies from training data for each vector type
vectBinCount.fit(x_train)
vectCount.fit(x_train)
vectTfidf.fit(x_train)

# transform training data
X_train_binCount = vectBinCount.transform(x_train)
X_train_count = vectCount.transform(x_train)
X_train_tfidf = vectTfidf.transform(x_train)

# transform validation data
X_validate_binCount = vectBinCount.transform(x_valid)
X_validate_count = vectCount.transform(x_valid)
X_validate_tfidf = vectTfidf.transform(x_valid)

# create and train Logistic Regression model for each type of vectorizer
print("Training models...")
models = []
model_binCount = MultinomialNB()
models.append(model_binCount)
model_binCount.fit(X_train_binCount, y_train)

model_count = MultinomialNB()
models.append(model_count)
model_count.fit(X_train_count, y_train)

model_tfidf = MultinomialNB()
models.append(model_tfidf)
model_tfidf.fit(X_train_tfidf, y_train)

# run models on validation set
print("Running models...")
predictions_binCount = model_binCount.predict(X_validate_binCount)
predictions_count = model_count.predict(X_validate_count)
predictions_tfidf = model_tfidf.predict(X_validate_tfidf)

# print model performance
accuracies = []
accuracy_binCount = metrics.accuracy_score(y_valid, predictions_binCount)
accuracy_count = metrics.accuracy_score(y_valid, predictions_count)
accuracy_tfidf = metrics.accuracy_score(y_valid, predictions_tfidf)

accuracies.append(accuracy_binCount)
accuracies.append(accuracy_count)
accuracies.append(accuracy_tfidf)

print("Naive Bayes Performance using binary counts :")
print(accuracy_binCount)
print("Naive Bayes Performance using counts :")
print(accuracy_count)
print("Naive Bayes Performance using TF-IDF :")
print(accuracy_tfidf)

saveModel(model_count, isFineGrainedMode, "NB")

print(predictions_count)
