# COMP 551 Mini Project 4
2019-04-17
Segev, Michael
Jacquier, Pierre
Han, Zhenze

Models and experiments are split in seperate python scripts that 
all use common classes to load files and save models to file.

1. NaiveBayesBench.py
	Run this file to test different feature extraction pipelines 
	with a NB classifier on Stanford Sentiment Treebank.

2. SVMBench.py
	Run this file to train and test/validate Support Vector Machine 
	model on Stanford Sentiment Treebank.

3. RNNBench.py
	Run this file to train and test/validate Recursive Neural 
	Network model on Stanford Sentiment Treebank.

4. DecisionTreesBench.py
	Run this file to train and test/validate extremely random trees 
	model on Stanford Sentiment Treebank.

5. metaClassifier.py
	Run this file to train and test/validate stacking ensemble 
	meta-classifier on on Stanford Sentiment Treebank using 
	pre-trained models saved as pickle files.

## Library Dependecies:
	re
	numpy
	scikit-learn
	keras
	tensorflow-gpu

## Copyright
[MIT license](LICENSE.md)