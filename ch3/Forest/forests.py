from sklearn import datasets
from sklearn.cross_validation import train_test_split

from sklearn.ensemble import RandomForestClassifier
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

# n_estimators : number of tree in the forest
# n_jobs : number of jobs to run in parallel for both fit and predict.
# if -1, then n_jobs = number of cores
forest = RandomForestClassifier(criterion="entropy",
								n_estimators=10,
								random_state=1,
								n_jobs=2)

# vstack : Stack arrays in sequence vertically (row wise). (row concatenation)
# hstack : Stack arrays in sequence horizontally (column wise). (col concatenation)
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

forest.fit(X_train, y_train)
plot_decision_regions(X_combined, y_combined,
					  classifier=forest, test_idx=range(105,150))

plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend(loc='upper left')
plt.show()