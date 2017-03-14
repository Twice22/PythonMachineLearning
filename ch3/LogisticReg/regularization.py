from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from plottingDecisionRegions import * # homemade file

import numpy as np

# iris datasets in scikit-learn library (150 samples)
# very widely used datasets !
iris = datasets.load_iris()
X = iris.data[:, [2, 3]] # we only use the 2nd and 3rd features

# vector containing [Iris-Setosa, ... Iris-Versicolor, ... Iris-Virginica]
# Note : Iris-Setosa... are already stored as integers !
# use np.unique(y) to convince yourself. Should return (0, 1, 2)
y = iris.target


# we split the dataset into training and test datasets
# using sklearn.cross_validation module
X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=0.3, random_state=0) 
# 30 % test data (45 samples)
# 70 % training data (150 - 45) samples


# features scaling using sklearn.preprocessing module
# to enhance performance
sc = StandardScaler()

# the fit method from StandardScaler estimates the parameters
# µ (mean) and sig (standard deviation) and standardized the
# training data
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


# vstack : Stack arrays in sequence vertically (row wise). (row concatenation)
# hstack : Stack arrays in sequence horizontally (column wise). (col concatenation)
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

# C : Inverse of regularization strength; must be a positive float.
# C = 1/lamba where lambda is the regularization param in logistic reg.
# random_state : The seed of the pseudo random number generator to use when shuffling the data.
weights, params = [], []
for c in np.arange(-5, 5):
	lr = LogisticRegression(C=10**c, random_state=0)
	lr.fit(X_train_std, y_train)
	# Coefficient of the features in the decision function.
	# we only collected the weight coef of the class 2 vs all classifier
	weights.append(lr.coef_[1]) 
	params.append(10**c)
weights = np.array(weights)
plt.plot(params, weights[:,0], label="petal length")
plt.plot(params, weights[:, 1], linestyle='--', label='petal width')
plt.xlabel('C')
plt.ylabel('weight coeeficient')
plt.legend(loc='upper left')
plt.xscale('log') # log scale for x axis
plt.show()
