from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
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


# plot de decisions regions of our newly trained percetron model
# using plot_decision_regions from plottingDecisionRegions

# vstack : Stack arrays in sequence vertically (row wise). (row concatenation)
# hstack : Stack arrays in sequence horizontally (column wise). (col concatenation)
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

# we use the rbf kernel (Gaussian kernel) see p77
# small value for γ (gamma), so decision boundary = soft
svm1 = SVC(kernel='rbf', random_state=0, gamma=0.20, C=1).fit(X_train_std, y_train)
plot_decision_regions(X_combined_std, y_combined, classifier=svm1, test_idx=range(105,150))

plt.xlabel('petal length [standardized]')
plt.ylabel('petl width [standardized]')
plt.legend(loc='upper left')
plt.show()

# bigger value for γ (gamma), so decision boundary much tighter
svm2 = SVC(kernel='rbf', random_state=0, gamma=100, C=1).fit(X_train_std, y_train)
plot_decision_regions(X_combined_std, y_combined, classifier=svm2, test_idx=range(105,150))

plt.xlabel('petal length [standardized]')
plt.ylabel('petl width [standardized]')
plt.legend(loc='upper left')
plt.show()