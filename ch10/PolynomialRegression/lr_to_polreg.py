# in the previous sections, we assumed a linear relationship between
# explanatory and response variables but we can have sth like :
# y = w0 + w1x + w2x²x² + ... + wdx^d


# we will use PolynomialFeatures transformer class from scikit
# to add a quadratic term (d = 2) to a simple reg problem with
# one explanatory variable and compare the pol to the linear fit

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


# np.newaxis transform it to a col vect
X = np.array([258.0, 270.0, 294.0, 320.0, 342.0, 368.0,
			  396.0, 446.0, 480.0, 586.0])[:, np.newaxis]

y = np.array([236.4, 234.4, 252.8, 298.6, 314.2, 342.2,
			  360.8, 368.0, 391.2, 390.8])

lr = LinearRegression()
pr = LinearRegression()

# add a second degreee polynomial term
quadratic = PolynomialFeatures(degree=2)
X_quad = quadratic.fit_transform(X)

# fit a simple linear reg model for comparison:
lr.fit(X, y)

# [[250] [260] [270] ... [590]]
X_fit = np.arange(250,600,10)[:, np.newaxis]
y_lin_fit = lr.predict(X_fit)


# fit a multiple reg model on the transformed features for
# polynomial regression:
pr.fit(X_quad, y)

# need to transform to pol before predicting using pr classifier
y_quad_fit = pr.predict(quadratic.fit_transform(X_fit))
plt.scatter(X, y, label='training points') # nuage de point
plt.plot(X_fit, y_lin_fit, label='linear fit', linestyle='--')
plt.plot(X_fit, y_quad_fit, label='quadratic fit')
plt.legend(loc='upper left')
plt.show()

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

y_lin_pred = lr.predict(X)
y_quad_pred = pr.predict(X_quad)
print('Training MSE linear: %.3f, quadractic: %.3f' % (
	mean_squared_error(y, y_lin_pred),
	mean_squared_error(y, y_quad_pred)))
print('Training R^2 linear: %.3f, quadratic: %.3f' % (
	r2_score(y, y_lin_pred),
	r2_score(y, y_quad_pred)))