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


# Learning curve start here
import matplotlib.pyplot as plt
import numpy as np
from sklearn.learning_curve import learning_curve

# penalty is the norm used in penalization when used regularized lr
pipe_lr = Pipeline([
			('scl', StandardScaler()),
			('clf', LogisticRegression(
						penalty='l2', random_state=0))
			])

# np.linspace(0.1, 1.0, 10)
# Returns num evenly spaced samples over the interval [0.1, 1.0]
# the return will be an array with 10 values (last param) so :
# [0.1 0.2 0.3 ... 1.0]

# cv stands for cross-validation, has we've specified a number
# we will use a Stratified 10-fold here !

# train_sizes inside learning_curve() :
# number of training samples used to generate the learning
# curves. Here we will use 10 evenly spaced relative intervals
# for the training set sizes.

# to have a smoother curve just change the last number in
# the linspace function. i.e : np.linspace(0.1, 1.0, 20)

train_sizes, train_scores, test_scores = learning_curve(estimator=pipe_lr,
						   X=X_train,
						   y=y_train,
						   train_sizes=np.linspace(0.1, 1.0, 10),
						   cv=10,
						   n_jobs=1) # -1 to use all the CPU of the computer

print(train_sizes)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(train_sizes, train_mean, color='b', marker='o',
								  markersize=5,
								  label='training accuracy')
plt.fill_between(train_sizes,
				 train_mean + train_std,
				 train_mean - train_std,
				 alpha=0.15, color='b')

plt.plot(train_sizes, test_mean, color='g', marker='s',
								  markersize=5, linestyle='--',
								  label='validation accuracy')
plt.fill_between(train_sizes,
				 test_mean + test_std,
				 test_mean - test_std,
				 alpha=0.15, color='g')

plt.grid()
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.8, 1.0])

plt.show()

