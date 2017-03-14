# We are using the Wine dataset to see bagging in action.
# We only use the Wine classes 2 and 3 and we select only
# 2 features : Alcohol and Hue.

import pandas as pd
df_wine = pd.read_csv('wine.data', header=None)

df_wine.columns = ['Class label', 'Alcohol',
				   'Malic acid', 'Ash',
				   'Alcalinity of ash',
				   'Magnesium', 'Total phenols',
				   'Flavanoids', 'Nonflavanoid phenols',
				   'Proanthocyanins',
				   'Color intensity', 'Hue',
				   'OD280/OD315 of diluted wines',
				   'Proline']

# there are only 3 class label. We get ride of the first class
df_wine = df_wine[df_wine['Class label'] != 1]

y = df_wine['Class label'].values
X = df_wine[['Alcohol', 'Hue']].values

# we encode class labels into binary format and split into 60/40
# training/testing set
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split

le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y,
											test_size=0.40,
											random_state=1)

# we import BaggingClassifier from ensemble submodule
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(criterion='entropy',
							  max_depth=None,
							  random_state=1)

# we use a decision tree as the base classifier and create an
# ensemble of 500 decisions trees fitted on different bootstrap
# samples of the training dataset :
bag = BaggingClassifier(base_estimator=tree,
						n_estimators=500,
						max_samples=1.0, # The number of samples to draw from X to train each base estimator.
						max_features=1.0, # The number of features to draw from X to train each base estimator.
						bootstrap=True, # wether samples are drawn with replacement
						bootstrap_features=False, # wether features are drawn with replacement
						n_jobs=1, # nb of core to use from the GPU
						random_state=1)

# here max_samples=1.0 (float) so we draw max_samples * X.shape[0] samples.
# idem max_features=1.0 (float) so we draw max_features * X.shape[1] features.

# calculate accuracy score of the prediction on the training and
# test dataset to compare the perf of the bagging classifier to
# the performance of the single unpruned decision tree:
from sklearn.metrics import accuracy_score
tree = tree.fit(X_train, y_train)

y_train_pred = tree.predict(X_train)
y_test_pred = tree.predict(X_test)

tree_train = accuracy_score(y_train, y_train_pred)
tree_test = accuracy_score(y_test, y_test_pred)

print('Decision tree train/test accuracies %.3f/%.3f'
		% (tree_train, tree_test)) # we get an high overfit on the test dataset


# training the baggingClassifier
bag = bag.fit(X_train, y_train)

y_train_pred = bag.predict(X_train)
y_test_pred = bag.predict(X_test)

bag_train = accuracy_score(y_train, y_train_pred)
bag_test = accuracy_score(y_test, y_test_pred)

print('Bagging train/test accuracies %.3f/%.3f'
		% (bag_train, bag_test)) # overfit less


import numpy as np
import matplotlib.pyplot as plt
# Next compare the decision regions between the decision tree
# and the bagging classifier:

# use to know the limit of the graph to display
x_min = X_train[:, 0].min() - 1 # min of feature 0 (1)
x_max = X_train[:, 0].max() + 1
y_min = X_train[:, 1].min() - 1
y_max = X_train[:, 1].max() + 1 # max of feature 1 (2)

# np.arange create a array from x_min to x_max each 0.1 step
# explanation of meshgrid in decision_region.py
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
					 np.arange(y_min, y_max, 0.1))

# see decision_region for more explanation
f, axarr = plt.subplots(nrows=1, ncols=2,
						sharex='col',
						sharey='row',
						figsize=(16,6))

for idx, clf, tt in zip([0, 1], [tree, bag],
						['Decision tree', 'Bagging']):
	clf.fit(X_train, y_train)

	Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
	Z = Z.reshape(xx.shape)

	axarr[idx].contourf(xx, yy, Z, alpha=0.3)

	axarr[idx].scatter(X_train[y_train==0, 0],
					   X_train[y_train==0, 1],
					   c='blue', marker='^')

	axarr[idx].scatter(X_train[y_train==1, 0],
					   X_train[y_train==1, 1],
					   c='red', marker='o')

	axarr[idx].set_title(tt)

axarr[0].set_ylabel('Alcohol', fontsize=12)
plt.text(10.2, -1.2,
		 s='Hue',
		 ha='center', va='center', fontsize=12)

plt.show()