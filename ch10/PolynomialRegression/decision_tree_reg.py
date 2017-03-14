import pandas as pd
import matplotlib.pyplot as plt

# the separator between values here is not the traditional ';' but \s+
df = pd.read_csv('housing.data', header=None, sep='\s+')

df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM',
			  'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B',
			  'LSTAT', 'MEDV']

X = df[['LSTAT']].values # return a col vector [[], [], .. []]
y = df['MEDV'].values

from sklearn.tree import DecisionTreeRegressor

tree = DecisionTreeRegressor(max_depth=3)

# train our tree
tree.fit(X, y)

# flatten() transform to 1D-array (X vect col)
# argsort() returns the inices that would sort an array
sort_idx = X.flatten().argsort()

def lin_regplot(X, y, model):
	plt.scatter(X, y, c='blue')
	plt.plot(X, model.predict(X), color='red')
	return None

lin_regplot(X[sort_idx], y[sort_idx], tree)

plt.xlabel('% lower status of the population [LSTAT]')
plt.ylabel('Price in $1000\'s [MEDV]')
plt.show()