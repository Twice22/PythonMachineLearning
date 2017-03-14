import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler

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

# 2 Step :
# A variance matrix looks like this : (3 features here)
#     | σ1² σ12 σ13 |
# Σ = | σ21 σ2² σ23 |
#     | σ31 σ32 σ3² |
# The eigenvectors of the covariance matrix represent the
# principal components (the dir of max variance), whereas the
# corresponding eigenvalues define their magnitude. For the wine
# they are 13 features so we'll obtain 13x13 dim cov and 13
# eigenvectors and eigenvalues

# A eigenvalue v satisfies Σv = λv
# we'll use linalg.eig from Numpy to obtain the eigenpairs

cov_mat = np.cov(X_train_std.T) # compute cov matrice

# 3 Step : retrieves eigenpairs
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat) # get eigenpairs
print('\nEigenvalues \n%s' % eigen_vals)

# we want the top k eigenvectors based on their corresponding
# eigenvalues because it corresponds to the greatest variance
# (most information).

# Additional Step : select k eigenvectors (here we print a graph
# to show the importance of each eigenvalues )
# first let's plot the variance explained ratio : λ/Σλ
tot = sum(eigen_vals) # Σλ
# sorted(eigen_vals, reverse=True) # decreasing value of eigenvalues
# var_exp contains decreasing value of variance explained ratio.
var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]

# cumulative sum
# a = np.array([[1,2,3], [4,5,6]])
# np.cumsum(a)
# output : array([ 1,  3,  6, 10, 15, 21])
cum_var_exp = np.cumsum(var_exp)

plt.bar(range(1,14), var_exp, alpha=0.5, align='center',
		label='individual explained variance')
plt.step(range(1,14), cum_var_exp, where='mid',
		label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.show()

# we saw that the first 2 components explain almost 60% of the variance
# Note : it is an unsupervised learning algorithm, so the information
# about the class labels is IGNORED here

# 4 Step : select k eigenvectors.
# We start by sorting eigenparis by decreasing order of eigenvalues
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:,i])
				for i in range(len(eigen_vals))]
eigen_pairs.sort(reverse=True)

# 4 & 5 Step : select 2 eigenvectors + create matrice
# we choose k = 2 eigenvectors that captures 60 percent of the
# variance in this dataset. We choose 2 because we want to plot it
# eigen_pairs[0][1] : select eigen_vec from first (0) eigen_pairs
# [:, np.newaxis] : transpose the row to col
# np.hstack : concatenate the cols
w = np.hstack((eigen_pairs[0][1][:, np.newaxis],
			   eigen_pairs[1][1][:, np.newaxis]))
print('Matrix W:\n', w)


# 6 Step : we can obtain the new 2-dim subspcae : x' = xW
# here x is a 1*13-dim row vector, W is 13x2 and so x' is 1*2 :
X_train_pca = X_train_std.dot(w)

# visualize Wine training set, now stored as a 124*2-dim matrice
# 178 samples - (30% of testing)*178 = 124
colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']
for l, c, m in zip(np.unique(y_train), colors, markers):
	print(y_train)
	plt.scatter(X_train_pca[y_train==l, 0],
				X_train_pca[y_train==l, 1],
				c=c, label=l, marker=m)

plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()
