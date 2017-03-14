from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

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


# train the Perceptron model using sklearn.linear_model module
ppn = Perceptron(n_iter=40, eta0=0.1, random_state=0) # initialize param. eta0 = ƞ
ppn.fit(X_train_std, y_train) # train


# so we know have estimated all the weights : w
# having trained a model, we can now make prediction
y_pred = ppn.predict(X_test_std)
# print misclassified samples
print('Misclassifed samples: %d' % (y_test != y_pred).sum())
# print accuracy : 1 - misclassified samples
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

# plot de decisions regions of our newly trained percetron model
# using plot_decision_regions from plottingDecisionRegions

# vstack : Stack arrays in sequence vertically (row wise). (row concatenation)
# hstack : Stack arrays in sequence horizontally (column wise). (col concatenation)
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X=X_combined_std,
					  y=y_combined,
					  classifier=ppn,
					  test_idx=range(105,150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()