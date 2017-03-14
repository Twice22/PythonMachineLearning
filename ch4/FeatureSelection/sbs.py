# sbs is not implemented in scikit-learn yet. So let's do it

from sklearn.base import clone
from itertools import combinations

import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score



class SBS():
	def __init__(self, estimator, k_features,
		scoring=accuracy_score,
		test_size=0.25, random_state=1):
		self.scoring = scoring
		self.estimator = clone(estimator)
		self.k_features = k_features
		self.test_size = test_size
		self.random_state = random_state

	def fit(self, X, y):
		# split between training and testing set
		X_train, X_test, y_train, y_test = train_test_split(X,
			y, test_size=self.test_size, random_state=self.random_state)

		dim = X_train.shape[1] # nb of col of X_train (nb of features)
		self.indices_ = tuple(range(dim)) # (0, 1, 2, ..., dim)
		self.subsets_ = [self.indices_] # [(0, 1, 2, ... dim)]
		score = self._calc_score(X_train, y_train, X_test,
								 y_test, self.indices_) #fct at the bottom
		self.scores_ = [score]

		while dim > self.k_features:
			scores = []
			subsets = []

			# combinations('ABCD', 2) --> AB AC AD BC BD CD
			# combinations(range(4), 3) --> 012 013 023 123
			for p in combinations(self.indices_, r=dim-1):
				#print(p)
				score = self._calc_score(X_train, y_train, X_test,
										 y_test, p)
				scores.append(score)
				subsets.append(p)

			# return indice of the max from the scores array
			best = np.argmax(scores)
			# get the tuple of features that give the best score
			self.indices_ = subsets[best]
			# keep a trace of each remaining features at each iterations
			self.subsets_.append(self.indices_)
			dim -= 1

			# keep a trace of the score after removing 1 feature
			self.scores_.append(scores[best])
		# save the last score
		self.k_score_ = self.scores_[-1]

		return self

	def transform(self, X):
		return X[:, self.indices_]

	def _calc_score(self, X_train, y_train, X_test, y_test, indices):
		# calc score using training and testing set
		# and the indices of the features remaining

		self.estimator.fit(X_train[:, indices], y_train)
		y_pred = self.estimator.predict(X_test[:, indices])
		score = self.scoring(y_test, y_pred)
		return score