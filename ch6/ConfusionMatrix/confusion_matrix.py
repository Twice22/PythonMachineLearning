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
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline


# Grid search start here
import matplotlib.pyplot as plt
from sklearn.grid_search import GridSearchCV

# pipeline : Standardize the data (same scale)
# then apply SVC (Support Vector Classification)
pipe_svc = Pipeline([('scl', StandardScaler()),
					 ('clf', SVC(random_state=1))])

# training the learning algorithm
pipe_svc.fit(X_train, y_train)

# use the trained algorithm to know to predict class of testing
# dataset
y_pred = pipe_svc.predict(X_test)

# draw confusion matrix with y_true being y_test (real data from dataset)
# and y_pred being the predicted data using X_test and the trained algorithm
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
print(confmat) # 2x2 matrix here

# draw the matrix using plot :
fig, ax = plt.subplots(figsize=(2.5, 2.5))
# Display an array as a matrix in a new figure window.
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]): # for all row
	for j in range(confmat.shape[1]): # for all col
		ax.text(x=j, y=i,
				s=confmat[i,j],
				va='center', ha='center')

plt.xlabel('predicted label')
plt.ylabel('true label')
plt.show()


# usefull metrics :
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, f1_score

print('Precision: %.3f' % precision_score(
					y_true=y_test, y_pred=y_pred))

print('Recall: %.3f' % recall_score(
					y_true=y_test, y_pred=y_pred))

print('F1: %.3f' % f1_score(
					y_true=y_test, y_pred=y_pred))

# Note : we can use different scoring metric other
# than accuracy in GridSearch via scoring parameter.

# positive class is the one labeled as class 1. To change
# this default behaviour. We can make use of make_scorer :

# from sklearn.metrics import make_scorer, f1_score
# scorer = make_scorer(f1_score, pos_label=0)
# gs = GridSearchCV(estimator=pipe_svc,
# 				  param_grid=param_grid,
# 				  scoring=scorer, # our scorer
#				  cv=10)