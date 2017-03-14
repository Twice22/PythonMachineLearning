# remember the ADaptive LInear NEuron (Adaline) from ch2.
# we used Gradient Descent and Stochastic Gradient Descent
# the cost fct in Adalin is the : Sum of Squared Errors (SSE)
# it's identical to the OLS (Ordinary Least Squares) function

#			J(w) = 1/2 * Σ(y(i) - ŷ(i))²

#			ŷ being the predicted value : ŷ = wTx

import numpy as np

class LinearRegressionGD(object):

	def __init__(self, eta=0.001, n_iter=20):
		self.eta = eta
		self.n_iter = n_iter

	def fit(self, X, y):
		# nb of weight = nb of col of X + 1 (w0)
		self.w_ = np.zeros(1 + X.shape[1])
		self.cost_ = []

		for i in range(self.n_iter):
			output = self.net_input(X) # ŷ (vector)
			errors = (y - output)
			self.w_[1:] += self.eta * X.T.dot(errors)
			self.w_[0] += self.eta * errors.sum()
			cost = (errors**2).sum() / 2.0 # J(w)
			self.cost_.append(cost)

		return self

	def net_input(self, X):
		return np.dot(X, self.w_[1:]) + self.w_[0]

	def predict(self, X):
		return self.net_input(X)