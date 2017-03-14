# we will use the Iris dataset from scikit-learn and we will
# only use 2 features (sepal width, petal length). And we will
# only use the classes : Iris-Versicolor and Iris-Virginica

from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

iris = datasets.load_iris()
# row 50 to the end, feature col 1 and 2
X, y = iris.data[50:, [1, 2]], iris.target[50:]

le = LabelEncoder()

# Fit label encoder and return encoded labels (array of int)
y = le.fit_transform(y)

# split iris into 50 % training/test data
X_train, X_test, y_train, y_test = train_test_split(X, y,
									test_size=0.5,
									random_state=1)

# we will train 3 different classifiers :
	# - logisitic regression
	# - decision tree
	# - k-nearest neighbors
# and look at their performances via a 10-fold cross-validation (cv)

from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

import numpy as np

# C is for the regularized LogisticRegression,
# penalty is the l2 norm for regularized LogisticReg
clf1 = LogisticRegression(penalty='l2', C=0.001, random_state=0)

clf2 = DecisionTreeClassifier(max_depth=1, criterion='entropy',
							  random_state=0)

# p is the param for the minkowki distance (see p95)
# other distance metric available here :
# http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.DistanceMetric.html
# n_neighbors : how many neighbors to use to classify the pt
clf3 = KNeighborsClassifier(n_neighbors=1, p=2, metric='minkowski')

pipe1 = Pipeline([['sc', StandardScaler()],
				  ['clf', clf1]])

# no new Standardization (dataset having same scale)
# for Decision Tree !!

pipe3 = Pipeline([['sc', StandardScaler()],
				  ['clf', clf3]])

clf_labels = ['Logistic Regression', 'Decision Tree', 'KNN']
print('10-fold cross validation:\n')
# for clf, label in zip([pipe1, clf2, pipe3], clf_labels):
# 	scores = cross_val_score(estimator=clf,
# 							 X=X_train,
# 							 y=y_train,
# 							 cv=10, # 10-fold cross-validation
# 							 scoring='roc_auc')
# 	print("ROC AUC: %0.2f (+/- %0.2f) [%s]"
# 			% (scores.mean(), scores.std(), label))

# combine the individual classifiers for majority rule voting
# using our MajorityVoteClassifier:
from majority_vote import MajorityVoteClassifier

mv_clf = MajorityVoteClassifier(classifiers=[pipe1, clf2, pipe3])
clf_labels += ['Majority Voting']
all_clf = [pipe1, clf2, pipe3, mv_clf]
for clf, label in zip(all_clf, clf_labels):
	scores = cross_val_score(estimator=clf,
							 X=X_train,
							 y=y_train,
							 cv=10,
							 scoring='roc_auc')
	print("ROC AUC: %0.2f (+/- %0.2f) [%s]"
			% (scores.mean(), scores.std(), label))


# compute ROC curves from test to check if the MajorityVoteClassifier
# generalizes well to unseen data.
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

colors = ['black', 'orange', 'blue', 'green']
linestyles = [':', '--', '-.', '-']


# if script launchs from itself execute the following code
if __name__ == "__main__":
	# compute auc for each classifier
	for clf, label, clr, ls \
			in zip(all_clf, clf_labels, colors, linestyles):

		# assuming the label of the positive class is 1
		y_pred = clf.fit(X_train, y_train).predict_proba(X_test)[:, 1]
		
		# y_true=y_train[test] : true binary label in range [0,1]
		# if labels not binary, use pos_label in roc_curve

		# y_score=probas[:, 1] :  target scores, prob estimates
		# of the positive class

		# pos_label : int : label considered as positive, others
		# are considered negative

		# fpr : increasing false + rates such taht elt i is the false
		# positive rate of predictions with score >= thresholds[i]

		# tpr : idem with true positive rate

		# thresholds : decreasing thresholds on the decision fct used
		# to compute fpr and tpr.
		fpr, tpr, thresholds = roc_curve(y_true=y_test,
										 y_score=y_pred)
		# compute area under the curve (auc)
		roc_auc = auc(x=fpr, y=tpr)
		plt.plot(fpr, tpr,
				 color=clr,
				 linestyle=ls,
				 label='%s (auc = %0.02f)' % (label, roc_auc))

	plt.legend(loc='lower right')

	# plot best performance line (top left corner)
	plt.plot([0, 1], [0, 1],
			 linestyle='--',
			 color='gray',
			 linewidth=2)
	plt.xlim([-0.1, 1.1])
	plt.ylim([-0.1, 1.1])
	plt.grid()
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')

	plt.show()