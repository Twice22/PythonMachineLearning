# to install libraries
# install anacoda or 
# $ python3 -m pip install lib_name

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('iris.data', header=None)
#print(df.tail())

#       0    1    2    3               4
# 145  6.7  3.0  5.2  2.3  Iris-virginica
# 146  6.3  2.5  5.0  1.9  Iris-virginica
# 147  6.5  3.0  5.2  2.0  Iris-virginica
# 148  6.2  3.4  5.4  2.3  Iris-virginica
# 149  5.9  3.0  5.1  1.8  Iris-virginica

# extract 4 feature column (class-label)
# first 50 are Iris-setosa
# last 50 are Iris-Versicolor

y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1) # ternaire on a vector

# extract 0 feature column (sepal length)
# and 2 feature column (petal length)
# and assign then to a matrix X

X = df.iloc[0:100, [0, 2]].values

# following lines won't be executed if this file is
# imported from another Python file
if __name__ == "__main__":
	plt.scatter(X[:50,0], X[:50, 1], color='red', marker='o', label='setosa')
	plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
	plt.xlabel('petal length')
	plt.ylabel('sepal length')
	plt.legend(loc='upper left')
	plt.show()



