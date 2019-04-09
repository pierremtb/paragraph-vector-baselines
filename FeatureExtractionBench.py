from ClassifierDataPrepper import ClassifierDataPrepper
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn import metrics

dataPath = "./stanfordSentimentTreebank/"

print("Opening files...")
cdp = ClassifierDataPrepper(dataPath)

Xl, Yl = cdp.getXYlabeledBinary()

# Split labelled data into train and validation sets
X_train, X_validate, Y_train, Y_validate = train_test_split(Xl, Yl, test_size=0.2, random_state=101)

# Building different vectorizers used to parse the text into features
print("Extracting features from data ...")
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
model_binCount = BernoulliNB()  # LogisticRegression()
model_binCount.fit(X_train_binCount, Y_train)

model_count = BernoulliNB()  # LogisticRegression()
model_count.fit(X_train_count, Y_train)

model_tfidf = BernoulliNB()  # LogisticRegression()
model_tfidf.fit(X_train_count, Y_train)

# run models on validation set
print("Running models...")
predictions_binCount = model_binCount.predict(X_validate_binCount)
predictions_count = model_count.predict(X_validate_count)
predictions_tfidf = model_tfidf.predict(X_validate_tfidf)

# print model performance
accuracy_binCount = metrics.accuracy_score(Y_validate, predictions_binCount)
accuracy_count = metrics.accuracy_score(Y_validate, predictions_count)
accuracy_tfidf = metrics.accuracy_score(Y_validate, predictions_tfidf)

print("Naive Bayes Performance using binary counts :")
print(accuracy_binCount)
print("Naive Bayes Performance using counts :")
print(accuracy_count)
print("Naive Bayes Performance using TF-IDF :")
print(accuracy_tfidf)

