from ClassifierDataPrepper import ClassifierDataPrepper
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.svm import SVC
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical

dataPath = "./stanfordSentimentTreebank/"

print("Opening files...")
cdp = ClassifierDataPrepper(dataPath)

# x_train, y_train, x_valid, y_valid, x_test, y_test = cdp.getXYlabeledBinary()  # best score is 0.7733031674208145
x_train, y_train, x_valid, y_valid, x_test, y_test = cdp.getXYlabeledSplit()  # best score is 0.39547511312217193

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

embed_dim = 128
lstm_out = 300
batch_size= 32

##Buidling the LSTM network
model = Sequential()
# model.add(Embedding(2500, embed_dim,input_length = len(X_train_tfidf), dropout=0.1))
model.add(LSTM(lstm_out, dropout_U=0.1, dropout_W=0.1))
model.add(Dense(2,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])

#Here we train the Network.

model.fit(X_train_tfidf, y_train, batch_size =batch_size, nb_epoch = 1,  verbose = 5)

# Measuring score and accuracy on validation set

score,acc = model.evaluate(X_validate_tfidf, y_valid, verbose = 2, batch_size = batch_size)
print("Logloss score: %.2f" % (score))
print("Validation set Accuracy: %.2f" % (acc))

# model_binCount = SVC(kernel=kernel, gamma=gamma, C=C, decision_function_shape=decision_shape)
# model_count = SVC(kernel=kernel, gamma=gamma, C=C, decision_function_shape=decision_shape)
# model_tfidf = SVC(kernel=kernel, gamma=gamma, C=C, decision_function_shape=decision_shape)

# model_binCount.fit(X_train_binCount, y_train)
# model_count.fit(X_train_count, y_train)
# model_tfidf.fit(X_train_tfidf, y_train)

# # run models on validation set
# print("Running models...")
# predictions_binCount = model_binCount.predict(X_validate_binCount)
# predictions_count = model_count.predict(X_validate_count)
# predictions_tfidf = model_tfidf.predict(X_validate_tfidf)

# # print model performance
# accuracy_binCount = metrics.accuracy_score(y_valid, predictions_binCount)
# accuracy_count = metrics.accuracy_score(y_valid, predictions_count)
# accuracy_tfidf = metrics.accuracy_score(y_valid, predictions_tfidf)

# print("SVM Performance using binary counts :")
# print(accuracy_binCount)
# print("SVM Performance using counts :")
# print(accuracy_count)
# print("SVM Performance using TF-IDF :")
# print(accuracy_tfidf)