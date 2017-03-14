# we read the wine dataset (178 wine, 13 features).
# it is available : https://archive.ics.uci.edu/ml/datasets/Wine

import pandas as pd
import numpy as np

# to split the sample into training and testing dataset
from sklearn.cross_validation import train_test_split

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

# 3 classes : 1, 2, 3 that corresponds to different types of graphs
print('Class labels', np.unique(df_wine['Class label']))
df_wine.head()

# .iloc is primarily INTEGER position based
# .loc is primarily LABEL based
# X => matrice without first column ; y => first col vector (classes)
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)