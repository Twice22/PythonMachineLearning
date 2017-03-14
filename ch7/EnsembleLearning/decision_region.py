from majority_vote_example import *

sc = StandardScaler()

X_train_std = sc.fit_transform(X_train)

from itertools import product

all_clf = [pipe1, clf2, pipe3, mv_clf]

# select min and max of x and y to delimited the boundary 
# of the graph
x_min = X_train_std[:, 0].min() - 1
x_max = X_train_std[:, 0].max() + 1
y_min = X_train_std[:, 1].min() - 1
y_max = X_train_std[:, 1].max() + 1

# Note : we standardize alos the Decision Tree because we gonna
# display decision boundary for all classifier and we want to
# have similar data so we can compare the graphs

# np.arange create a array from x_min to x_max each 0.1 step
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
					 np.arange(y_min, y_max, 0.1))

# xx : here a 62x57 matrix with the same row being
# [x_min, x_min+0.1, ..., ... x_max]

## idem for yy but for the y axis. yy same dimension as xx

# axarr is an array of axis object
# f is the figure
f, axarr = plt.subplots(nrows=2, ncols=2, # 2 graphs by rows/cols
						sharex='col', # each subplot col will share a X axis
						sharey='row', # each subplot row will share a Y axis
						figsize=(14,10))

# product('ABCD', 'xy') --> Ax Ay Bx By Cx Cy Dx Dy
# product(range(2), repeat=3) --> 000 001 010 011 100 101 110 111
# here : product([0, 1], [0, 1]) --> [(0, 0), (0, 1), (1, 0), (1, 1)]
for idx, clf, tt in zip(product([0, 1], [0, 1]),
						all_clf, clf_labels):

	# train each classifier
	clf.fit(X_train_std, y_train)

	# numpy.c : Translates slice objects to concatenation along the second axis.
	# np.c_[np.array([[1,2,3]]), 0, 0, np.array([[4,5,6]])]
	# return : array([[1, 2, 3, 0, 0, 4, 5, 6]])

	# ravel : Return a contiguous flattened array.
	# x = np.array([[1, 2, 3], [4, 5, 6]])
	# return : [1 2 3 4 5 6]

	# we actually predict class label on every single pt of the
	# meshgrid !!!
	# it return an 1D array : shape = [n_samples]
	Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
	# reshape the prediction so it will be the same size
	# as the xx matrix : 62*57
	# note : size of Z before must be equal to 62*57...
	Z = Z.reshape(xx.shape)

	axarr[idx[0],idx[1]].contourf(xx, yy, Z, alpha=0.3)

	# scatter plot of class 0 (y_train==0) using 2 features (0 and 1)
	axarr[idx[0], idx[1]].scatter(X_train_std[y_train==0, 0],
								  X_train_std[y_train==0, 1],
								  c='blue',
								  marker='^',
								  s=50)
	# scatter plot of class 0 (y_train==0) using 2 features (0 and 1)
	axarr[idx[0], idx[1]].scatter(X_train_std[y_train==1, 0],
								  X_train_std[y_train==1, 1],
								  c='red',
								  marker='o',
								  s=50)
	axarr[idx[0], idx[1]].set_title(tt)

plt.text(-4, -4,
		 s='Sepal width [standardized]',
		 ha='center', va='center', fontsize=12)
plt.text(-12, 4,
		 s='Petal length [standardized]',
		 ha='center', va='center',
		 fontsize=12, rotation=90)

plt.show()


# Note : to access individual parameter inside a GridSearch obj
# we use .get_params() that we defined in marority_vote.py :
# mv.clf_get_params()