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


def save(filename):
	pass


def main():
	pass


if __name__=="__main__":
	train_text, train_labels = load("train.csv")
	test_text, test_labels = load("test.csv")

	vect = TfidfVectorizer(min_df=0.001, ngram_range=(1, 1))
	vect.fit(np.hstack((train_text, test_text)))
	train_text = vect.transform(train_text)
	test_text = vect.transform(test_text)

	model = lm.LogisticRegression(penalty='l2')
	scores = cv.cross_val_score(model, train_text, train_labels, cv=5)