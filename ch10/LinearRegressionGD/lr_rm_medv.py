# we will use our class from linear_regression.py to do a linear
# regression between RM (number of rooms) variable and MEDV (the
# housing prices). we will standardize the var to better convergence
# of the Gradient Descent algorithm.
import pandas as pd
import matplotlib.pyplot as plt

# the separator between values here is not the traditional ';' but \s+
df = pd.read_csv('housing.data', header=None, sep='\s+')

df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM',
			  'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B',
			  'LSTAT', 'MEDV']


X = df[['RM']].values # return an col vector (matrix on n*1)
y = df['MEDV'].values # return an array of values

from sklearn.preprocessing import StandardScaler
from linear_regression import LinearRegressionGD

# initialize 2 standardization classes
sc_x = StandardScaler()
sc_y = StandardScaler()

# standardize X and y ( x - mean)/deviation
X_std = sc_x.fit_transform(X)
y_std = sc_y.fit_transform(y)

lr = LinearRegressionGD() # initialize our LinearRegressionGD
lr.fit(X_std, y_std) # train our lr to get weights


# plot the cost against the number of epochs (iterations)
# to check if the linear regression has converged :
plt.plot(range(1, lr.n_iter+1), lr.cost_)
plt.ylabel('SSE')
plt.xlabel('Epoch')
plt.show()


# then visualize of well the linear regression line fits the
# training data. We define a fct that plot scatterplot + the
# regression line:

def lin_regplot(X, y, model):
	plt.scatter(X, y, c='blue')
	plt.plot(X, model.predict(X), color='red')
	return None

# we use lin_regplot fct to plot the number of rooms against house
# prices :
lin_regplot(X_std, y_std, lr)
plt.xlabel('Average number of rooms [RM] (standardized)')
plt.ylabel('Price in $1000\'s [MEDV] (standardized)')
plt.show()


# we see a interesting line : y =3 that show that the house prices
# have been clipped

# also we can apply the inverse_transform method of the StandarScaler
# to show the real price of house in 1k$ (for 5 rooms):
num_rooms_std = sc_x.transform([5.0])
price_std = lr.predict(num_rooms_std)
print("Price in $1000's: %.3f" %\
	  sc_y.inverse_transform(price_std))


# print('Slope: %.3f' % lr.w_[1])

# note : as we are working with standardized variables, w0 = 0,
# the plot intercept the y axis in 0 !
# print('Intercept: %.3f' % lr.w_[0])


# use scikit-learn's LinearRegression object that makes use of
# the LIBLINEAR library :
from sklearn.linear_model import LinearRegression
slr = LinearRegression()
slr.fit(X, y) # train without standardization needed
print('Slope: %.3f' % slr.coef_[0])
print('Intercept: %.3f' % slr.intercept_)

# we see that lr fitted with unstandardized RM and MEDV var
# yielded different model coefficients. Let's compare to our GD :
lin_regplot(X, y, slr) # homemade fct
plt.xlabel('Average number of rooms [RM] (standardized)')
plt.ylabel('Price in $1000\'s [MEDV] (standardized)')

# the overall result of this new plot looks identical to our
# GD implementation
plt.show()