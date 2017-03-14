# we gonna use the RANSAC algorithm from scikit-learn
# see the txt file for the explanation

import pandas as pd
import matplotlib.pyplot as plt

# the separator between values here is not the traditional ';' but \s+
df = pd.read_csv('housing.data', header=None, sep='\s+')

df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM',
			  'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B',
			  'LSTAT', 'MEDV']


X = df[['RM']].values # return an col vector (matrix on n*1)
y = df['MEDV'].values # return an array of values

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor

# we wrap our linear model in the RANSAC
# min_samples : min nb of the randomly chosen samples
# residual_metric : we use a callable lambda fct that simply
# calculates the absolute vertical distances between the fitted
# line and the sample points.
# residual_threshold : we only allowed samples to be included in
# the inlier set if their vertical distance to the fitted line is
# within 5 distance units. (difficult to find a good one see p292)
ransac = RANSACRegressor(LinearRegression(),
				max_trials=100, # max iterations
				min_samples=50,
				residual_metric=lambda x: np.sum(np.abs(x), axis=1),
				residual_threshold=5.0,
				random_state=0)

# train the ransac classifier
ransac.fit(X, y)

# obtain the inliers and outliers from the fitted RANSAC linear
# regression model and plot them together with the linear fit:

# Boolean mask of inliers classified as True, that is to say :
# an array of boolean corresponding to the pts that are inliers :
# True if inliers, False otherwise :
# [True True False True False True True ...]
inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask) # outlier mask
line_X = np.arange(3, 10, 1) # [3, 4, 5, ... 9]

# line_X[:, np.newaxis] : [[3] [4] [5]] (vector col)
# line_y_ransac is the predicted price for each house with different
# nb of rooms represented by line_X
line_y_ransac = ransac.predict(line_X[:, np.newaxis])

# take only inlier_mask from X (inlier_mask is an array of boolean)
plt.scatter(X[inlier_mask], y[inlier_mask], c='blue',
			marker='o', label='Inliers')

plt.scatter(X[outlier_mask], y[outlier_mask], c='lightgreen',
			marker='s', label='Outliers')

plt.plot(line_X, line_y_ransac, color='red')

plt.xlabel('Average number of rooms [RM]')
plt.ylabel('Price in $1000\'s [MEDV]')
plt.legend(loc='upper left')
plt.show()

# print('Slope: %.3f' % ransac.estimator_.coef_[0])
# print('Intercept: %.3f' % ransac.estimator_.intercept_)
