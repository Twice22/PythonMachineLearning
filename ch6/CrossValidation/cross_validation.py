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


# We know have to standardize the datasets (features data having
# the same scale). And we want to apply PCA to reduce the dim from
# 30 to 32 so we can visualize. Instead of doing those 3 things
# separately, (on test train dataset then test dataset)...
# we can chain StandardScaler, PCA, LogisticRegression via pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


# Grid search start here
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.cross_validation import cross_val_score
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# pipeline : Standardize the data (same scale)
# then apply SVC (Support Vector Classification)
pipe_svc = Pipeline([('scl', StandardScaler()),
					 ('clf', SVC(random_state=1))])

# range of data of the hyperparameter C and gamma (see below)
param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

# Dictionary with parameters names (string) as keys and lists 
# of parameter settings to try as values, or a list of such
# dictionaries, in which case the grids spanned by each dictionary
# in the list are explored. This enables searching over
# any sequence of parameter settings.

# for linear_svm we only evaluate parameter C
# for rbf kernel SVM, we tuned C and gamma : k(x,x') = exp(-γ||x-x'||²)
param_grid = [{'clf__C': param_range,
			   'clf__kernel': ['linear']},
			  {'clf__C': param_range,
			   'clf__gamma': param_range,
			   'clf__kernel': ['rbf']}]


# needed because we use several core.
if __name__ == '__main__':
	# estimator : object type that implements fit and predict method
	gs = GridSearchCV(estimator=pipe_svc,
				  param_grid=param_grid,
				  scoring='accuracy',
				  cv=10, # 10-fold stratified
				  n_jobs=-1) # use all the core of the CPU

	# perform nested cross_validation :
	scores = cross_val_score(gs, X_train, y_train, scoring='accuracy', cv=5)
	print('CV accuracy: %.3f +/- %.3f' % (
				np.mean(score), np.std(scores)))

	# compare with DecisionTreeClassifier cv :
	gs = GridSearchCV(
			estimator=DecisionTreeClassifier(random_state=0),
			param_grid=[
				{'max_depth': [1, 2, 3, 4, 5, 6, 7, None]}],
			scoring='accuracy',
			cv=5)

	scores = cross_val_score(gs, X_train, y_train,
							 scoring='accuracy',
							 cv=5)

	print('CV accuracy: %.3f +/- %.3f' % (
				np.mean(scores), np.std(scores)))