# train decision tree with a max depth of 3 using
# entropy as a criterion for impurity.  feature scaling may
# be desired for visualization purposes but it is not a requirement
# for decision tree algorithms.

from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.tree import export_graphviz # allow to export the decision tree


from sklearn.tree import DecisionTreeClassifier
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

tree = DecisionTreeClassifier(criterion='entropy',
							  max_depth=3,
							  random_state=0)
tree.fit(X_train, y_train) # train

# vstack : Stack arrays in sequence vertically (row wise). (row concatenation)
# hstack : Stack arrays in sequence horizontally (column wise). (col concatenation)
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

# test_idx allows us to distinguish test_set from all set (X_combined & y_combined)
plot_decision_regions(X_combined, y_combined,
					  classifier=tree, test_idx=range(105,150))

plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc='upper left')
plt.show()


# export the decision tree as a .dot file
export_graphviz(tree, out_file='tree.dot',
				feature_names=['petal length', 'petal width'])
