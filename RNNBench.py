# COMP 551 Mini Project 4
# 2019-04-17
# Segev, Michael
# Jacquier, Pierre
# Han, Zhenze
# Run this file to train and test/validate Recursive Neural Network model on Stanford Sentiment Treebank

from ClassifierDataPrepper import ClassifierDataPrepper
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Conv1D, MaxPooling1D, Flatten, BatchNormalization, SpatialDropout1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from helpers import *

def labelToArray(label):
  arr = []
  for i in range(5):
    arr.append(int(label == i))
  return arr

def labelArrayToTensor(arr):
  t = np.zeros((len(arr), 5))
  for i, label in enumerate(arr):
    for j, binary in enumerate(labelToArray(label)):
      t[i, j] = binary
  return t    

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

max_features = 20000
maxlen = 267
batch_size = 16
embed_dim = 128

tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(x_train))

X_train = tokenizer.texts_to_sequences(x_train)
X_train = pad_sequences(X_train, maxlen=maxlen)
X_valid = tokenizer.texts_to_sequences(x_valid)
X_valid = pad_sequences(X_valid, maxlen=maxlen)
X_test = tokenizer.texts_to_sequences(x_test)
X_test = pad_sequences(X_test, maxlen=maxlen)

y_train = labelArrayToTensor(y_train)
y_valid = labelArrayToTensor(y_valid)
y_test = labelArrayToTensor(y_test)

print('x_train shape:', X_train.shape)
print('x_test shape:', X_valid.shape)
print('x_train shape:', X_train.shape)
print('x_test shape:', X_valid.shape)

print('Build model...')
model = Sequential()
model.add(Embedding(max_features, embed_dim, input_length=X_train.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(BatchNormalization())
# model.add(Dense(1, activation='sigmoid'))
# model.compile(loss='binary_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])
model.add(Dense(5, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=15,
          validation_data=(X_valid, y_valid))

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: {}".format(scores[1]*100))

models = []
models.append(model)
accuracies = []
accuracies.append(scores[1]*100)

saveBestModel(models, accuracies, isFineGrainedMode, "RNN")
