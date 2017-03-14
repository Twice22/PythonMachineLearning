# we will model the relationship between house prices and LSTAT
# (percent lower status of the population) using second degree and
# third degree polynomials and compare it to a linear fit.

import pandas as pd
import matplotlib.pyplot as plt

# the separator between values here is not the traditional ';' but \s+
df = pd.read_csv('housing.data', header=None, sep='\s+')

df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM',
			  'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B',
			  'LSTAT', 'MEDV']

X = df[['LSTAT']].values # return a col vector [[], [], .. []]
y = df['MEDV'].values


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# initialize LinearRegression (with param if use)
regr = LinearRegression()

# create polynomial features
# if input is 2 variables [a, b] :
# [1, a, b, a^2, ab, b^2]
quadratic = PolynomialFeatures(degree=2)
cubic = PolynomialFeatures(degree=3)

# example :
# x_test = quadratic.fit_transform([2, 5])
# print(x_test) : return [[1, 2, 5, 4, 10, 25]]

X_quad = quadratic.fit_transform(X)
X_cubic = cubic.fit_transform(X)

import numpy as np
from sklearn.metrics import r2_score
# linear fit
# create vector col from X.min() to X.max()-1 evenly spaced by 1
X_fit = np.arange(X.min(), X.max(), 1)[:, np.newaxis]
# train the Linear Regression
regr = regr.fit(X, y)
y_lin_fit = regr.predict(X_fit)
linear_r2 = r2_score(y, regr.predict(X)) # diff between real value and predicted one

# quadratric fit
regr = regr.fit(X_quad, y)
y_quad_fit = regr.predict(quadratic.fit_transform(X_fit))
quadratic_r2  = r2_score(y, regr.predict(X_quad))

# cubic fit
regr = regr.fit(X_cubic, y)
y_cubic_fit = regr.predict(cubic.fit_transform(X_fit))
cubic_r2 = r2_score(y, regr.predict(X_cubic))

# plot results
plt.scatter(X, y, label='training points', color='lightgray')
plt.plot(X_fit, y_lin_fit, label='linear (d=1), $R^2=%.2f$'
								% linear_r2, color='blue', lw=2,
								linestyle=':')
plt.plot(X_fit, y_quad_fit, label='quadratric (d=2), $R^2=%.2f$'
								% quadratic_r2,
								color='red',
								lw=2,
								linestyle='-')
plt.plot(X_fit, y_cubic_fit, label='cubic (d=3), $R^2=%.2f$'
								% cubic_r2,
								color='green',
								lw=2,
								linestyle='--')
plt.xlabel('% lower status of the population [LSTAT]')
plt.ylabel('Price in $1000\'s [MEDV]')
plt.legend(loc='upper right')
plt.show()

# R3 fit better (cubic) than the others but keep in mind that
# increasing the degree can lead to overfitting.

# besides, we can see, according the scatter plot (nuage de pts)
# that there seems to be a linear relationship between the log of
# MEDV and the square root of LSTAT. Let's see if that is correct :

# transform features
X_log = np.log(X)
y_sqrt = np.sqrt(y)

# fit_features
X_fit = np.arange(X_log.min()-1, X_log.max()+1, 1)[:, np.newaxis]

# train the linear regression
regr = regr.fit(X_log, y_sqrt)
y_lin_fit = regr.predict(X_fit)

# estimate r2 between real values and predicted one
linear_r2 = r2_score(y_sqrt, regr.predict(X_log))

# plot results
plt.scatter(X_log, y_sqrt, label='training points',
						   color='lightgray')
plt.plot(X_fit, y_lin_fit, label='linear (d=1), $R^2=%.2f$'
						   % linear_r2,
						   color='blue',
						   lw=2)

plt.xlabel('log(% lower status of the population [LSTAT])')
plt.ylabel('$\sqrt{Price \; in \; \$1000\'s [MEDV]}s')
plt.legend(loc='lower left')
plt.show()