# L1 regularization encourages sparsity. See explanation p115
# to use L1 regularization use penalty='l1' :

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sbs import * # our file

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

# Initialize sbs
knn = KNeighborsClassifier(n_neighbors=2)
sbs = SBS(knn, k_features=1)
sbs.fit(X_train_std, y_train)

# sbs object as know provided all his attributes with values.
# so let's plot the scores after we'd removed 1 feature

# sbs.subsets_ : [(0, 1, ... dim), (0, 1, 2, 3, 5, ..., dim), (...), (0,)]
k_feat = [len(k) for k in sbs.subsets_] # [dim, dim-1, ... 1]
plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylim([0.7, 1.1]) # y axi go from 0.7 to 1.1
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.show()


# we see that with 5 features we have 100 % correct classification
# let's see what are those 5 features :

k5 = list(sbs.subsets_[8]) # dim = 13 : 13 - 5 features = 8
print(df_wine.columns[1:][k5]) # 1 : column index without first row corresponding to 'Class Label'


# evaluate performance of the KNN classifier on the original test set :
knn.fit(X_train_std, y_train)
print('Training accuracy:', knn.score(X_train_std, y_train))
print('Test accuracy:', knn.score(X_test_std, y_test))

# training set : 98.4%, test set : 94.4 percent (slight overfitting ?)

# now the result using the selected 5-feature
knn.fit(X_train_std[:, k5], y_train) # k5 is a tuple
print('Training accuracy:', knn.score(X_train_std[:, k5], y_train))
print('Test accuracy:', knn.score(X_test_std[:, k5], y_test))