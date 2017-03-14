# BaseEstimator, ClassifierMixin are added to get some base
# functionality for free, including get_params et set_params
# to return the classifier's param and set it, as well as the
# score method to calculate the prediction accuracy

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import six # make it compatible for Python 2.7
from sklearn.base import clone
from sklearn.pipeline import _name_estimators

import numpy as np
import operator

class MajorityVoteClassifier(BaseEstimator, ClassifierMixin):
	""" A majority vote ensemble classifier

	Parameters
	----------
	classifiers : array-like, shape = [n_classifiers]
	Different classifiers for the ensemble

	vote : str, {'classlabel', 'probability'}
	  Default : 'classlabel'
	  If 'classlabel' the prediction is based on
	  the argmax of class labels. Else if
	  'probability', the argmax of the sum of
	  probabilities is used to predict the class
	  label (recommendd for calibrated classifiers).

	weights : array-like, shape = [n_classifiers]
	  Optional, default: None
	  If a list of 'int' or 'float' values are
	  provided, the classifiers are weighted by
	  importance; Uses uniform weights if 'weights=None'.

	"""

	def __init__(self, classifiers,
				 vote='classlabel', weights=None):
		self.classifiers = classifiers

		# generate a dict of estimators
		self.named_classifiers = {key: value for 
								  key, value in
								  _name_estimators(classifiers)}
		self.vote = vote
		self.weights = weights

	def fit(self, X, y):
		""" Fit classifiers.

		Parameters
		----------
		X : {array-like, sparse matrix},
			shape = [n_samples, n_features]
			Matrix of training samples.

		y : array-like, shape = [n_samples]
			Vector of target class labels.

		Returns
		-------
		self : object

		"""

		# Use LabelEncoder to ensure class labels start
		# with 0, which is important for np.argmax
		# call in self.predict
		self.lablenc_ = LabelEncoder()
		self.lablenc_.fit(y)
		self.classes_ = self.lablenc_.classes_ # array of unique class label (int)
		self.classifiers_ = []

		# fit X and y (already transformed and fitted) for each classifiers
		for clf in self.classifiers:
			fitted_clf = clone(clf).fit(X, self.lablenc_.transform(y))
			self.classifiers_.append(fitted_clf)
		return self

	def predict(self, X):
		""" Predict class labels for x.

		Parameters
		----------
		X : {array-like, sparse matrix}
			Shape = [n_samples, n_features]
			Matrix of training samples.

		Returns
		--------
		maj_vot : array-like, shape = [n_samples]
			Predicted class labels.

		"""

		if self.vote == 'probability':
			maj_vote = np.argmax(self.predict_proba(X), axis=1) # see fct below
		else: # 'classlabel' vote
			# Collect results from clf.predict calls
			predictions = np.asarray([clf.predict(X) for clf in
									  self.classifiers_]).T

			# http://docs.scipy.org/doc/numpy/reference/generated/numpy.apply_along_axis.html
			maj_vote = np.apply_along_axis(
									   lambda x:
									   np.argmax(np.bincount(x,
									   		weights=self.weights)),
									   axis=1,
									   arr=predictions)

		maj_vote = self.lablenc_.inverse_transform(maj_vote)
		return maj_vote

	def predict_proba(self, X):
		""" Predict class probabilities for X.

		Parameters
		----------
		X : {array-like, sparse matrix},
			shape = [n_samples, n_features]
			Training vectors, where n_samples is
			the number of samples and
			n_features is the number of features.

		Retuns
		-------
		avg_proba : array-like,
			shape = [n_samples, n_classes]
			Weighted average probability for
			each class per sample.

		"""

		# create an array of the proba of X for each classifiers
		# see majority_vote.txt
		probas = np.asarray([clf.predict_proba(X)
							for clf in self.classifiers_])
		avg_proba = np.average(probas, axis=0, weights=self.weights)
		return avg_proba

	def get_params(self, deep=True):
		""" get classifier parameter names for GridSearch """

		if not deep:
			return super(MajorityVoteClassifier,
						self).get_params(deep=False)
		else:
			out = self.named_classifiers.copy()
			for name, step in six.iteritems(self.named_classifiers):
				for key, value in six.iteritems(
						step.get_params(deep=True)):
					out['%s__%s' % (name, key)] = value
			return out
