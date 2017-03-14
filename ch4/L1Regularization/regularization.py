# L1 regularization encourages sparsity. See explanation p115
# to use L1 regularization use penalty='l1' :

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression

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

# 3 classes : 1, 2, 3 that corresponds to different types of grapes
print('Class labels', np.unique(df_wine['Class label']))
df_wine.head()

# .iloc is primarily INTEGER position based
# .loc is primarily LABEL based
# X => matrice without first column ; y => first col vector (classes)
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# standardized scalling
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)

# Initialize LogisticRegression with param (C = 1 /(2*lambda))
lr = LogisticRegression(penalty='l1', C=0.1)
lr.fit(X_train_std, y_train)

print('Training accuracy:', lr.score(X_train_std, y_train))
print('Testing accuracy:', lr.score(X_test_std, y_test))

# lr.intercept :
#	- the first value belongs to model that fits class 1 vs 2 & 3
#	- the 2nd value belongs to model that fits class 2 vs 1 & 3
#	- ...
lr.intercept_


# lr.coef_ :: display matrice of weights (here 3*13) :
# 13 weights for each class (3 classes in the wine dataset)
lr.coef_

# plot regularization path (weight coeff of the different
# features for different regularization strengths)
fig = plt.figure()
# hundred represent nrows, tens = ncols, unit = plot_number
# plot_number identify the particular subplot that this fct
# is create within the notional grid
ax = plt.subplot(111)
colors = ['blue', 'green', 'red', 'cyan',
		  'magenta', 'yellow', 'black', 'pink',
		  'lightgreen', 'lightblue', 'gray',
		  'indigo', 'orange']
weights, params = [], []
for c in np.arange(-4,6):
	lr = LogisticRegression(penalty='l1', C=10**c, random_state=0)
	lr.fit(X_train_std, y_train)
	weights.append(lr.coef_[1])
	params.append(10**c)
weights = np.array(weights)
# weights.shape[1] return number of col of weights mat (13 here bc 13 features)
for column, color in zip(range(weights.shape[1]), colors):
	plt.plot(params, weights[:, column],
		     label=df_wine.columns[column+1],
		     color=color)

plt.axhline(0, color='black', linestyle='--', linewidth=3)
plt.xlim([10**(-5), 10**5])
plt.ylabel('weight coeff')
plt.xlabel('C')
plt.xscale('log')
plt.legend(loc='upper left')
ax.legend(loc='upper center',
		  bbox_to_anchor=(1.38, 1.03),
		  ncol=1, fancybox=True)

plt.show()