import pandas as pd
import matplotlib.pyplot as plt

# the separator between values here is not the traditional ';' but \s+
df = pd.read_csv('housing.data', header=None, sep='\s+')

df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM',
			  'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B',
			  'LSTAT', 'MEDV']

X = df.iloc[:, :-1].values # take all features except MEDV
y = df['MEDV'].values

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,
									test_size=0.4,
									random_state=1)

import numpy as np
from sklearn.ensemble import RandomForestRegressor

forest = RandomForestRegressor(n_estimators=1000,
							   criterion='mse', #mse instead of entropy
							   random_state=1,
							   n_jobs=-1) # use all CPU core

forest.fit(X_train, y_train)
y_train_pred = forest.predict(X_train)
y_test_pred = forest.predict(X_test)

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

print('MSE train: %.3f, test: %.3f' % (
			mean_squared_error(y_train, y_train_pred),
			mean_squared_error(y_test, y_test_pred)))
print('R^2 train: %.3f, test: %.3f' % (
			r2_score(y_train, y_train_pred),
			r2_score(y_test, y_test_pred)))

# we see random forest tends to overfit the training data :
# R^2 train : 0.979 and R^2 test : 0.878

# let's plot de residuals of the prediction
# remember if the pt gather RANDOMly around 0 the algorithm works
# fine, but if there are a certain pattern (not random) we missed
# something (like a feature or so)

plt.scatter(y_train_pred, y_train_pred - y_train,
			c='black', marker='o', s=35, alpha=0.5,
			label="Training data")

plt.scatter(y_test_pred, y_test_pred - y_test,
			c='lightgreen', marker='s', s=35,
			alpha=0.7, label='Test data')

plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='red')
plt.xlim([-10, 50])
plt.show()