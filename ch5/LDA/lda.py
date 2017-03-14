import pandas as pd
import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

from plottingDecisionRegions import * # homemade file

# import Wine.data (13 features 178 samples)
df_wine = pd.read_csv('wine.data', header=None)

# X contains all rows without first col (col of the classes)
# y contains all rows of the first col (col of the classes : 3 different classes)
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y,
									test_size=0.3, random_state=0)

# 1 Step : standardize the training and testing dataset
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

# precision for floating point numbers, arrays and other np object (default : 8)
np.set_printoptions(precision=4)
mean_vecs = []

for label in range(1,4): # for label = 1, 2, 3 (3 classes)
	# X_train_std[y_train==label] select only the features vector
	# from class = label
	mean_vecs.append(np.mean(X_train_std[y_train==str(label)], axis=0))
	print('MV %s: %s\n' % (label, mean_vecs[label-1]))


# Step 3 : compute the within-class scatter matrix Sw : Sw = ΣSi
# where Si = Σ(x-mi)(x-mi)T is the individual scatter matrice of class i

# d = 13 # number of features
# S_W = np.zeros((d, d)) # 13x13 matrix of 0
# for label, mv in zip(range(1,4), mean_vecs):
 	# class_scatter = np.zeros((d, d))
 	# for row in X[y == str(label)]:
 		# reshape(m,n) reshape to a m row, n col matrix
 		# row is a col vector of the features of class i
 		# mv is a col vector of the mean of all vectors of class i
 		# row, mv = row.reshape(d, 1), mv.reshape(d, 1)
 		# class_scatter += (row-mv).dot((row-mv).T) # sum here : Si = Σ(x-mi)(x-mi)T 
 	# S_W += class_scatter # sum : Sw = Si
# print('Within-class scatter matrix: %sx%s' % (S_W.shape[0], S_W.shape[1]))

# we assume that the class labels in the training set are uniformly
# distributed. However that is not the case :
# bincount : Count number of occurrences of each value in array of non-negative ints.
# [1:] is used to avoid to retrieve the class label
# print('Class label distribution: %s' % np.bincount(y_train)[1:])


# Note : the covariance matrix is a normalized version of the
# scatter matrix (see p142) so it's better to compute it this way :
d = 13 # number of features
S_W = np.zeros((d,d))
for label, mv in zip(range(1,4), mean_vecs):
	class_scatter = np.cov(X_train_std[y_train==str(label)].T)
	S_W += class_scatter
print('Scaled within-class scatter matrix: %sx%s'
	% (S_W.shape[0], S_W.shape[1]))


# now we compute the between-class scatter matrix Sb :
# Sb = ΣNi(mi-m)(mi-m)T where m is the overall mean computed
# including samples from all classes
mean_overall = np.mean(X_train_std, axis=0)
d = 13 # number of features
S_B = np.zeros((d,d))
for i, mean_vec in enumerate(mean_vecs):
	# return nb of row of X having class y = str(i+1)
	# so return nb of features vector by class
	n = X[y==str(i+1), :].shape[0]
	mean_vec = mean_vec.reshape(d, 1) # make col vector
	mean_overall = mean_overall.reshape(d, 1) # make col vector
	S_B += n * (mean_vec - mean_overall).dot((mean_vec - mean_overall).T)
print('Between-class scatter matrix: %sx%s' 
	% (S_B.shape[0], S_B.shape[1]))

# Step 4 : get eigenpairs from Sw(-1)*Sb
# better to use np.linalg.eigh because Hermertian matrice
eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))

# Step 5 : select eigenvectors that corresponds to the k largest
# eigenvalue.

# we sort eigenvalues in descending order
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:,i])
				for i in range(len(eigen_vals))]
# key specify a function to be called on each list element
# prior to making comparisons. here the function return always
# the first elem of the comparison. It avoids error when 2 elts
# are identic
eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)
print('Eigenvalues in decreasing order:\n')
for eigen_val in eigen_pairs:
	print(eigen_val[0])

# measure how much of the class-discriminatory info is captured
# by the linear discriminants (eigenvectors) :
tot = sum(eigen_vals.real)
# eigenvalues (in %) sorted by decreasing order
discr = [(i/tot) for i in sorted(eigen_vals.real, reverse=True)]
cum_discr = np.cumsum(discr) # cumulative sum, see pca.py
plt.bar(range(1, 14), discr, alpha=0.5, align='center',
		label='individual "discriminability"')
plt.step(range(1,14), cum_discr, where='mid',
		label='cumulative "discriminability"')
plt.ylabel('"discriminability" ratio')
plt.xlabel('Linear Discriminants')
plt.ylim([-0.1, 1.1])
plt.legend(loc='best')
plt.show()

# stack the 2 most discriminative eigenvector col to create
# the transformation matrix W :
# eigen_pairs[0][1] : select eigen_vec from first (0) eigen_pairs
# [:, np.newaxis] : transpose the row to col and englobe the col
# so we have [[...]] instead of [...]
# np.hstack : concatenate the cols
w = np.hstack((eigen_pairs[0][1][:, np.newaxis].real,
			   eigen_pairs[1][1][:, np.newaxis].real))
print('Matrix W:\n', w)


# Step 6 : Project sample onto W : X' = XW
X_train_lda = X_train_std.dot(w)
colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']
for l, c, m in zip(np.unique(y_train), colors, markers):
	plt.scatter(X_train_lda[y_train==l, 0],
				X_train_lda[y_train==l, 1],
				c=c, label=1, marker=m)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='upper right')
plt.show()