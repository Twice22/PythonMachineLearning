from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

	# setup marker generator and color map
	markers = ('s', 'x', 'o', '^', 'v')
	colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
	cmap = ListedColormap(colors[:len(np.unique(y))]) # ListedColormap(colors[:1]) because 2 different value for y

	# >>> np.unique([1, 1, 2, 2, 3, 3])
	# array([1, 2, 3])


	# plot the decision surface
	# find min and max of the 2 features and add 1
	# so the max and mon features will be display within the
	# graph and won't be on the edge of the graph
	x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

	# create a pair of grid arrays xx1 and xx2
	# >>> np.arange(3,7,2) # from 3 to 7 each 2
	# array([3,5]) # array of evenly spaced values.
	xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
						   np.arange(x2_min, x2_max, resolution))

	# classifier=ppn see line 64
	Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)

	# ravel transform the matrix in an 1D-array
	# np.array concat the 2 arrays (2-dim array).
	# .T = transpose

	# reshape the predicted class labels Z into a grid with
	# the same dimensions as xx1 and xx2
	Z = Z.reshape(xx1.shape)

	# draw a contour plot via matplotlib's contourf
	plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
	plt.xlim(xx1.min(), xx1.max())
	plt.ylim(xx2.min(), xx2.max())

	# np.unique(y) return [0 1 2] (type of flower)
	# enumerate (np.unique(y)) return : 
	# (0, np.unique(y)(0)),(1, np.unique(y)(1)), (2, np.unique(y)(2))
	# plot class samples
	for idx, cl in enumerate(np.unique(y)):
		# make scatter plot (nuage de point) of x vs y
		plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
			alpha=0.8, c=cmap(idx),
			marker=markers[idx], label=cl)

	# highlight test samples
	if test_idx:
		X_test, y_test = X[test_idx, :], y[test_idx]
		plt.scatter(X_test[:, 0], X_test[:, 1], c='',
			alpha=1.0, linewidth=1, marker='o',
			s=55, label='test set')


if __name__ == "__main__":
	plot_decision_regions(X,y, classifier=ppn)
	plt.xlabel('sepal length [cm]')
	plt.ylabel('petal length [cm]')
	plt.legend(loc='upper left')

	plt.tight_layout()
	plt.show()