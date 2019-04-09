# Test Bench to compare different feature extraction pipelines
# COMP 551
# Boury Mbodj
# Humayun Khan
# Michael Segev
# Feb 14 2019
import ClassifierDataPrepper
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

dataPath = "./"
trainingDataPath = dataPath + "train/"
positiveTrainingDataPath = trainingDataPath + "pos/"
negativeTrainingDataPath = trainingDataPath + "neg/"
# testDataPath = dataPath + "test/"
testDataPath = None

print("Opening training and test files...")
cdp = ClassifierDataPrepper.ClassifierDataPrepper(positiveTrainingDataPath, negativeTrainingDataPath, testDataPath)

print("Preparing data frames...")
Xl, Yl = cdp.getXYlabeled()

# Split labelled data into train and validation sets
X_train, X_validate, Y_train, Y_validate = train_test_split(Xl, Yl, test_size=0.2, random_state=101)

# Building different vectorizers used to parse the text into features
print("Extracting features from data frames...")
vectBinCount = CountVectorizer(min_df=1, ngram_range=(1, 2), binary=True)
vectCount = CountVectorizer(min_df=1, ngram_range=(1, 2), binary=False)
vectTfidf = TfidfVectorizer(min_df=1, ngram_range=(1, 2))

# learn the vocabularies from training data for each vector type
vectBinCount.fit(X_train)
vectCount.fit(X_train)
vectTfidf.fit(X_train)

# transform training data
X_train_binCount = vectBinCount.transform(X_train)
X_train_count = vectCount.transform(X_train)
X_train_tfidf = vectTfidf.transform(X_train)

# transform validation data
X_validate_binCount = vectBinCount.transform(X_validate)
X_validate_count = vectCount.transform(X_validate)
X_validate_tfidf = vectTfidf.transform(X_validate)

# create and train Logistic Regression model for each type of vectorizer
print("Training models...")
logmodel_binCount = LogisticRegression()
logmodel_binCount.fit(X_train_binCount, Y_train)

logmodel_count = LogisticRegression()
logmodel_count.fit(X_train_count, Y_train)

logmodel_tfidf = LogisticRegression()
logmodel_tfidf.fit(X_train_count, Y_train)

# run models on validation set
print("Running models...")
predictions_binCount = logmodel_binCount.predict(X_validate_binCount)
predictions_count = logmodel_count.predict(X_validate_count)
predictions_tfidf = logmodel_tfidf.predict(X_validate_tfidf)

# print model performance
accuracy_binCount = metrics.accuracy_score(Y_validate, predictions_binCount)
accuracy_count = metrics.accuracy_score(Y_validate, predictions_count)
accuracy_tfidf = metrics.accuracy_score(Y_validate, predictions_tfidf)

print("Logistic Regression Performance using binary counts :")
print(accuracy_binCount)
print("Logistic Regression Performance using counts :")
print(accuracy_count)
print("Logistic Regression Performance using TF-IDF :")
print(accuracy_tfidf)

