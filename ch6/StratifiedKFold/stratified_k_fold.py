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


# We know have to standardize the datasets (features data having
# the same scale). And we want to apply PCA to reduce the dim from
# 30 to 32 so we can visualize. Instead of doing those 3 things
# separately, (on test train dataset then test dataset)...
# we can chain StandardScaler, PCA, LogisticRegression via pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

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

# using fit on pipe_lr, the transformer perfomrs fit, transform
# then pass the result to the other transformer and so on.
# when it finally hit the estimator it uses fit on the data
pipe_lr.fit(X_train, y_train)


# Stratified k-fold start here
import numpy as np
from sklearn.cross_validation import StratifiedKFold

# instanciate StratifiedKFold with 10subset (n_folds=10)
kfold = StratifiedKFold(y=y_train, n_folds=10, random_state=1)


scores = []

for k, (train, test) in enumerate(kfold):
	# train is the 9 fold, test in 1 fold.
	# in change 10 times (cause n_folds = 10)
	# test = [1, 2, 3, ..., 45] first iteration
	# train = [46, 47, ... , 450] first iteration

	# using the pipe, we are sure the samples in standardize
	# see previous pipeline, StandardScaler is used !
	pipe_lr.fit(X_train[train], y_train[train])
	score = pipe_lr.score(X_train[test], y_train[test])
	scores.append(score)
	print('Fold: %s, Class dist.: %s, Acc: %.3f' %(k+1,
								np.bincount(y_train[train]), score))

# mean/(standard deviation)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores),
								np.std(scores)))

# the previous was to illustrate how stratfied K fold works. but
# we can make use of  :

from sklearn.cross_validation import cross_val_score
scores = cross_val_score(estimator=pipe_lr, X=X_train,
						 y=y_train,
						 cv=10,
						 n_jobs=1) # parallel treatment
print('CV accuracy scores: %s' % scores)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores),
								np.std(scores)))