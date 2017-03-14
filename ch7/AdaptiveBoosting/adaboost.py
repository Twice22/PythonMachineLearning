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

# we use the wine subset to train an AdaBoost ensemble classifier
# via the base_estimator attribute, we will train AdaBoostClassifier
# on 500 decision tree stumps
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score

tree = DecisionTreeClassifier(criterion='entropy', # The function to measure the quality of a split.
							  max_depth=1)

ada = AdaBoostClassifier(base_estimator=tree,
						 n_estimators=500,
						 learning_rate=0.1,
						 random_state=0)

#  fit the data to our Decision tree to train it
tree = tree.fit(X_train, y_train)

# predict what will be the class of de pt from X_train and X_test
y_train_pred = tree.predict(X_train)
y_test_pred = tree.predict(X_test)

# measure the accuracy (how well it classifies the newly pt)
tree_train = accuracy_score(y_train, y_train_pred)
tree_test = accuracy_score(y_test, y_test_pred)

print('Decision tree train/test accuracies %.3f/%.3f'
		% (tree_train, tree_test)) 
# we see Decision tree stump seems to overfit the training data
# in contrast with the unpruned decision tree that we saw in
# bagging.py

ada = ada.fit(X_train, y_train)

y_train_pred = ada.predict(X_train)
y_test_pred = ada.predict(X_test)

ada_train = accuracy_score(y_train, y_train_pred)
ada_test = accuracy_score(y_test, y_test_pred)

print('AdaBoost train/test accuracies %.3f/%.3f'
			% (ada_train, ada_test))


# we see that using AdaBoost classifier we could have increased
# th accuracy_score. However we should note that it is a bad
# practice to select a model based on the repeated usage of
# the test set (see ch6)

# Check what's the decision regions look like :
import numpy as np
import matplotlib.pyplot as plt

# see decision_region for explanation if needed
x_min = X_train[:, 0].min() - 1;
x_max = X_train[:, 0].max() + 1;
y_min = X_train[:, 1].min() - 1;
y_max = X_train[:, 1].max() + 1;

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
					 np.arange(y_min, y_max, 0.1))

f, axarr = plt.subplots(1, 2,
						sharex='col',
						sharey='row',
						figsize=(8,3))

for idx, clf, tt in zip([0, 1], [tree, ada],
						['Decision tree', 'AdaBoost']):
	clf.fit(X_train, y_train)
	Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
	Z = Z.reshape(xx.shape)
	
	axarr[idx].contourf(xx, yy, Z, alpha=0.3)
	axarr[idx].scatter(X_train[y_train==0, 0],
					   X_train[y_train==0, 1],
					   c='blue',
					   marker='^')

	axarr[idx].scatter(X_train[y_train==1, 0],
					   X_train[y_train==1, 1],
					   c='red',
					   marker='o')

axarr[idx].set_title(tt)
axarr[0].set_ylabel('Alcohol', fontsize=12)

plt.text(10.2, -1.2,
		 s='Hue',
		 ha='center',
		 va='center',
		 fontsize=12)

plt.show()