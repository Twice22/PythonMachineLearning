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
lr = LogisticRegression(C=1000.0, random_state=0)
lr.fit(X_train_std, y_train)
plot_decision_regions(X=X_combined_std,
					 y=y_combined,
					 classifier=lr,
					test_idx=range(105,150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()

# predict_proba our trained LogisticRegression :
# example : predict proba of the first Iris-Setosa sample :
lr.predict_proba(X_test_std[0,:]);
# return : array([[ 0.000, 0.063, 0.937]])
# it shos that the model predicts a chance of 93.7 that
# the sample is a iris-Virginica and 6.3% Iris-Versicolor