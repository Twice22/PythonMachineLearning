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

# then using pipe_lr with the test dataset perform here
# transform then transform then predict...
# see p172 for a picture
print('Test Accuracy: %.3f' % pipe_lr.score(X_test, y_test))