# 100 samples assigned the class 1 and 100 other assigned to class -1
# using the xor gate :
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)
# 200 rows x 2 col matrice with random value from univariate
# gaussian distribution of mean 0 and variance 1
X_xor = np.random.randn(200, 2) 

# >>> np.logical_xor([True, True, False, False], [True, False, True, False])
# output : array([False,  True,  True, False], dtype=bool)
y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)
# transform the preceding true-false matrice in a matrice of -1/1 value
y_xor = np.where(y_xor, 1, -1)

if __name__ == "__main__":
	plt.scatter(X_xor[y_xor==1, 0], X_xor[y_xor==1, 1],
		c='b', marker='x', label='1')
	plt.scatter(X_xor[y_xor==-1, 0], X_xor[y_xor==-1, 1],
		c='r', marker='s', label='-1')
	plt.ylim(-3.0)
	plt.legend()
	plt.show()

# we cannot separate those values with linear hyperplane.
# to deal with those kind of data we have to project it onto
# a new higher-dimension hyperplane (see p76)