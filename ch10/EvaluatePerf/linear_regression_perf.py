# we gonna evaluate the performance of the linear regression models
# using all the features of the dataset this time !


import pandas as pd
import matplotlib.pyplot as plt

# the separator between values here is not the traditional ';' but \s+
df = pd.read_csv('housing.data', header=None, sep='\s+')

df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM',
			  'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B',
			  'LSTAT', 'MEDV']

from sklearn.cross_validation import train_test_split

X = df.iloc[:, :-1].values # all feature but last (MEDV)
y = df['MEDV'].values # MEDV will represent the y (medv is the price of house in the dataset)

import numpy as np
from sklearn.linear_model import LinearRegression

X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=0.3, random_state=0)

# initialize LinearRegression classifier
slr = LinearRegression()

# train the linear regression
slr.fit(X_train, y_train)

# predict the value of the linear regression for the training dataset
y_train_pred = slr.predict(X_train)

# predict the value of the test dataset (yet unseen data)
y_test_pred = slr.predict(X_test)


# we cannot visualize the linear regression line (we are not in 2d
# or 3d : 13 features here...)
# we gonna use the residual plots that are a commonly used graphical
# analysis for diagnosing regression models to detect nonlinearity


# we plot a residual plot where we simply subtract the true target
# variables from our predicted responses:
plt.scatter(y_train_pred, y_train_pred - y_train, c='blue',
			marker='o', label='Training data')

plt.scatter(y_test_pred, y_test_pred - y_test, c='lightgreen',
			marker='s', label='Test data')

plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')

# horizontal lines at -10 and 50 weight of line = 2
plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='red')
plt.xlim([-10, 50])
plt.show()

# in the case of a perfect prediction, the residuals would be
# exactly zero.
# for a good regression model, we would expect that the errors
# are RANDOMLY distributed and the residual should be RANDOMLY
# scattered around the centerline.
# if we see patterns, it means that our model is unable to capture
# some explanatory information


# another useful qunatitative measure of a model's perf is the
# so-called Mean Squared Error (MSE) : 1/2 * Σ(y(i) - ŷ(i))²
from sklearn.metrics import mean_squared_error
print('MSE train: %.3f, test: %.3f' % (
	mean_squared_error(y_train, y_train_pred),
	mean_squared_error(y_test, y_test_pred)))

# MSE on training : 19.96
# MSE of test : 27.20
# so our model is overfitting the training data

# we can use R² (see p296 explanation)
# if R² = 1, the model fits the data perfectly with a MSE = 0
from sklearn.metrics import r2_score
print('R^2 train: %.3f, test: %.3f' % 
			(r2_score(y_train, y_train_pred),
			 r2_score(y_test, y_test_pred)))
