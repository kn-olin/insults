import pandas as pd
import numpy as np
import nltk
import string

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn import cross_validation as cv
from sklearn import metrics

import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import sklearn.naive_bayes as nb


def loadTrain(filename):
	"""Load comment and whether it is an insult from csv."""
	data = pd.read_csv(filename)
	text = data["Comment"].values
	insult = data["Insult"].values
	return text, insult

def loadTest(filename):
	"""Load comment and id."""
	data = pd.read_csv(filename)
	text = data["Comment"].values
	comment_id = data["id"].values
	return text, comment_id

def tokenizeText():
	"""Tokenize a full text block."""
	words = data["Comment"].apply(nltk.word_tokenize)


def makeFeatures(text):
	pass


def save(filename, data):
	pass

def main_logistic_regression(train_text, train_label, test_text, verbose=True):
	''' logistic regression '''
	clf = lm.LogisticRegression(penalty='l2')

	scores = cv.cross_val_score(clf, train_text, train_label, cv=5, scoring='roc_auc')
	mean_score = np.mean(scores)

	clf.fit(train_text, train_label)

	if verbose:
		print("auc: %0.5f" % mean_score)
		results = pd.DataFrame({'features': vect.get_feature_names(), 'weights': clf.coef_[0]})
		results = results.sort('weights', ascending=False)
		print('\n###### insult')
		print(results[:10])
		print('\n###### not insult')
		print(results[-10:])

	return clf, mean_score

def main_naive_bayes(train_text, train_label, test_text, verbose=True):
	''' naive_bayes '''
	clf = nb.MultinomialNB(alpha=0.4, fit_prior=True)

	scores = cv.cross_val_score(clf, train_text, train_label, cv=5, scoring='roc_auc')
	mean_score = np.mean(scores)

	clf.fit(train_text, train_label)

	if verbose:
		print("auc: %0.5f" % mean_score)
		results = pd.DataFrame({'features': vect.get_feature_names(), 'weights': clf.coef_[0]})
		results = results.sort('weights', ascending=False)
		print('\n###### insult')
		print(results[:25])

	return clf, mean_score


def savePrediction(filename, test_id, test_text, clf):
	''' output '''
	prob = clf.predict_proba(test_text)[:,1]
	predict = pd.DataFrame({"Id": test_id, "Insult": prob})
	predict.to_csv(filename, index=False)


if __name__=="__main__":
	train_text, train_label = loadTrain("train.csv")
	test_text, test_id = loadTest("test.csv")

	# vect = CountVectorizer(ngram_range=(1,1))
	vect = TfidfVectorizer(min_df=0.001, ngram_range=(1, 1))
	vect.fit(np.hstack((train_text, test_text)))
	train_text = vect.transform(train_text)
	test_text = vect.transform(test_text)
	
	clf, mean_score = main_naive_bayes(train_text, train_label, test_text)

	savePrediction("scrap.csv", test_id, test_text, clf)