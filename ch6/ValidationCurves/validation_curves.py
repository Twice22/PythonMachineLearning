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

# penalty is the norm used in penalization when used regularized lr
pipe_lr = Pipeline([
			('scl', StandardScaler()),
			('clf', LogisticRegression(
						penalty='l2', random_state=0))
			])


# validation_curve start here
import matplotlib.pyplot as plt
import numpy as np
from sklearn.learning_curve import validation_curve

param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

# validation_curve use stratified k-fold cv by default
# to estimate the perf of the model. Here we use a 10-fold (cv=10)

# we specified de param that we wanted to evaluate in this
# validation curve. In this case we are evaluating C, which is
# the inverse regularization param of the LogisticRegression.
# param_name='clf_C'
# use estimator.get_params().keys() to see the param_name
# available

# we varie the value of the param C using param_range. here
# we are evaluating the validation curve for C taking value in
# param_range array

# estimator : object type that implements fit and predict method

# print(pipe_lr.get_params().keys())
train_scores, test_scores = validation_curve(
							estimator=pipe_lr,
							X=X_train,
							y=y_train,
							param_name='clf__C',
							param_range=param_range,
							cv=10)

train_mean = np.mean(train_scores, axis=1)
# standard deviation of training sample
train_std = np.std(train_scores, axis=1)

test_mean = np.mean(test_scores, axis=1)
# standard deviation of testing sample
test_std = np.std(test_scores, axis=1)

plt.plot(param_range, train_mean,
		 color='b', marker='o',
		 markersize=5,
		 label='training accuracy')
plt.fill_between(param_range, train_mean + train_std,
				 train_mean - train_std, alpha=0.15,
				 color='b')

plt.plot(param_range, test_mean, color='g',
		 linestyle='--', marker='s', markersize=5,
		 label='validation accuracy')
plt.fill_between(param_range,
				 test_mean + test_std,
				 test_mean - test_std,
				 alpha=0.15,
				 color='g')

plt.grid()
plt.xscale('log') # cf param_range log scale !
plt.legend(loc='lower right')
plt.xlabel('Parameter C')
plt.ylabel('Accuracy')
plt.ylim([0.8, 1.0])
plt.show()