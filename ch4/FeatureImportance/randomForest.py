# we will train a forest of 10 000 trees on the Wine (13 features)
# we don't need to normalize or standardize the data as we are
# dealing with a tree

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier

df_wine = pd.read_csv('wine.data', header=None)
df_wine.columns = ['Class label', 'Alcohol',
				   'Malic acid', 'Ash',
				   'Alcalinity of ash', 'Magnesium',
				   'Total phenols', 'Flavanoids',
				   'Nonflavanoid phenols',
				   'Proanthocyanins',
				   'Color intensity', 'Hue',
				   'OD280/OD315 of diluted wines',
				   'Proline']

# .iloc is primarily INTEGER position based
# .loc is primarily LABEL based
# X => matrice without first column ; y => first col vector (classes)
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# standardized scalling
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)

feat_labels = df_wine.columns[1:] # label (1: to avoid title : 'Class Label')

# n_jobs = -1 => use all the cores of the computer
forest = RandomForestClassifier(n_estimators=10000,
								random_state=0,
								n_jobs=-1)
forest.fit(X_train, y_train)
importances = forest.feature_importances_ # collect feature importance
indices = np.argsort(importances)[::-1]
for f in range(X_train.shape[1]): # for each feature
	print("%2d) %-*s %f" % (f+1, 30, feat_labels[f],
									importances[indices[f]]))

# print
plt.title('Feature Importances')
plt.bar(range(X_train.shape[1]),
		importances[indices],
		color='lightblue',
		align='center')
plt.xticks(range(X_train.shape[1]), feat_labels, rotation=90) # rotation of the text
plt.xlim([1, X_train.shape[1]]) #x axis from 1 to nb of feature (nb of col)
plt.tight_layout()
plt.show()

# we can use a threshold with transform method to only select the
# feature that are greater than this threshold :
# X_selected = forest.transform(X_train, threshold=0.15)
# X_selected.shape # return (124, 3)