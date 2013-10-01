import pandas as pd
import numpy as np
import nltk
import string

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn import cross_validation as cv
from sklearn import metrics

import matplotlib.pyplot as plt
import sklearn.linear_model as lm


def load(filename):
	"""Load comment and whether it is an insult from csv."""
	data = pd.read_csv(filename)
	text = data["Comment"].values
	try:
		insult = data["Insult"].values
	except KeyError:
		insult = []
	return text, insult


def tokenizeText():
	"""Tokenize a full text block."""
	words = data["Comment"].apply(nltk.word_tokenize)


def makeFeatures(text):
	pass


def save(filename, data):
	pass


if __name__=="__main__":
	train_text, train_labels = load("train.csv")
	test_text, test_labels = load("test.csv")

	vect = TfidfVectorizer(min_df=0.001, ngram_range=(1, 1))
	vect.fit(np.hstack((train_text, test_text)))
	train_text = vect.transform(train_text)
	test_text = vect.transform(test_text)

	clf = lm.LogisticRegression(penalty='l1')

	scores = cv.cross_val_score(clf, train_text, train_labels, cv=5)
	print("\naccuracy: %0.5f" % np.mean(scores))

	clf.fit(train_text, train_labels)
	results = pd.DataFrame({'features': vect.get_feature_names(), 
		                  'weights': clf.coef_[0]})
	results = results.sort('weights', ascending=False)
	print('\n###### insult')
	print(results[:10])
	print('\n###### not insult')
	print(results[-10:])

	# pred = clf.predict(train_text)
	pred = clf.predict(test_text)