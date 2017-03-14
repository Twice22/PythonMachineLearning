# Tuning hyperparameterS via grid search
# Grid search allow to find the best combination of several
# hyperparameters. To do so, the algorithm use brute force.

# we will use the Breast Cancer Wisconsin dataset.
# it is like :
# |   1   |   2   |   3   |   ...   |  32   |
# |  ID   |  M/B  |       |         |       |

# they are 32 col. first 2 cols contains :
#		- unique ID numbers
#		- corresponding diagnosis (M=malignant, B=benign)

import pandas as pd

# read the csv file (datasets)
df = pd.read_csv('wdbc.data', header=None)

# assign the 30 features to a Numpy array x.
# We use LabelEncoder to transform the class labels from their
# original string representation (M/B) to integer (remember ch3/4)
from sklearn.preprocessing import LabelEncoder

# loc retrieve label (string) values
# iloc retrieve integer values !

# retrieve all rows and 30 features other than first 2
X = df.loc[:, 2:].values

# retrieve all features corresponding to the diagnosis (M/B)
y = df.loc[:, 1].values

# intanciate LabelEncoder obj
le = LabelEncoder()

# Fit label encoder and return encoded labels (array of int)
# M = 1 and B = 0 afterwards
y = le.fit_transform(y)

# divide between training and test dataset
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
									test_size=0.20, random_state=1)


# starting confusion matrix code
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


# Grid search start here
import matplotlib.pyplot as plt

# Pipeline obj task a lIST of TUPLES as input, where first value
# in each tuple is an arbitrary id string that we can use to
# access the individual elt in the pipeline, and second elt is
# as scikit-learn transformer or estimator (class intanciation)
pipe_lr = Pipeline([('scl', StandardScaler()),
					('pca', PCA(n_components=2)),
					('clf', LogisticRegression(random_state=1))])


# intermediate steps in a pipeline is scikit-learn transformers.
# last step is an estimator.
# here we have 2 transformers : StandardScaler, PCA
# and 1 estimator : LogisticRegression

# using fit on pipe_lr, the transformer performs fit, transform
# then pass the result to the other transformer and so on.
# when it finally hits the estimator it uses fit on the data
pipe_lr.fit(X_train, y_train)


# begin the treatment of the ROC curves
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from scipy import interp
import numpy as np

# we are only feature 4 and 14 of all samples (all row) :
X_train2 = X_train[:, [4, 14]]

# cross validation with 3-Fold stratified
cv = StratifiedKFold(y_train, n_folds=3,
					 random_state=1)

fig = plt.figure(figsize=(14,10))
mean_tpr = 0.0
# create array from 0 to 1 with 100 pts evenly-spaced
mean_fpr = np.linspace(0, 1, 100)
all_tpr = []

for i, (train, test) in enumerate(cv):
	# train is the 2 fold length, test in 1 fold.
	# it's change 10 times (cause n_folds = 3)
	# test = [1, 2, 3, ..., 149] first iteration
	# train = [150, 152, ... , 454] first iteration

	# using the pipe, we are sure the samples is standardize
	# see previous pipeline, StandardScaler is used !
	probas = pipe_lr.fit(X_train2[train],
				y_train[train]).predict_proba(X_train2[test])

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
	fpr, tpr, thresholds = roc_curve(y_train[test],
									 probas[:, 1],
									 pos_label=1)

	# interpolation to construct discrete data pt for the curve
	# return an array on discrete data pt to plot the curve
	mean_tpr += interp(mean_fpr, fpr, tpr)

	mean_tpr[0] = 0.0
	# Compute Area Under the Curve (AUC)
	# fpr is x axis, tpr is y axis
	roc_auc = auc(fpr, tpr)
	plt.plot(fpr, tpr, lw=1,
			 label='ROC fold %d (area = %0.2f)'
			 % (i+1, roc_auc))

# plot diagonal : correspond to random guessing
plt.plot([0, 1], [0, 1], linestyle='--',
		 color=(0.6, 0.6, 0.6),
		 label='random guessing')

# compute the mean of the 3 preceding fold curve
mean_tpr /= len(cv)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, 'k--',
		 label='mean ROC (area = %0.2f)' % mean_auc, lw=2)

# plot perfect performance : tpr = 1, fpr = 0 (top left corner)
plt.plot([0, 0, 1],
		 [0, 1, 1],
		 lw=2,
		 linestyle=':',
		 color='black',
		 label='perfect performance')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.title('Receiver Operator Characteristic')
plt.legend(loc='lower right')
plt.show()

# if we are just interested in the ROC AUC score, we could
# also use roc_auc_score from sklearn.metrics. Example after
# fitting it on the two-feature training set:
# Note :
# pipe_svc is not defined here, see grid_search.py for i.e

# pipe_svc = pipe_svc.fit(x_train2, y_train)
# y_pred2 = pipe_svc.predict(X_test[:, [4, 14]])

# from sklearn.metrics import roc_auc_score
# from sklearn.metrics import accuracy_score
# print('ROC AUC: %.3f' % roc_auc_score(
# 			y_true=y_test, y_score=y_prd2))
# print('Accuracy: %.3f' % accuracy_score(
# 			y_true=y_test, y_pred=y_pred2))
