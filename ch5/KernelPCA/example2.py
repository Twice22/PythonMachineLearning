# separating concentric circles

from sklearn.datasets import make_circles
from kernelPCA import * # homemade file

import matplotlib.pyplot as plt

# factor : scale factor between inner and outer circle (def : 0.8)
X, y = make_circles(n_samples=1000, random_state=123,
					noise=0.1, factor=0.2)

plt.scatter(X[y==0, 0], X[y==0, 1], color='red', marker='^', alpha=0.5)
plt.scatter(X[y==1, 0], X[y==1, 1], color='blue', marker='o', alpha=0.5)
plt.show()


# standard PCA approach (not linearly separable, won't work)
from sklearn.decomposition import PCA

# initialize param for PCA
scikit_pca = PCA(n_components=2)

# fit the model with X and apply the dimensionality reduction on X
X_spca = scikit_pca.fit_transform(X)

# number of rows of the subplot grid
# number of col of the subplot grid
# tuple of integers with width and height in inches
# return :
#	- a tuple containing figure and axes object(s)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14,6))

# first figure
ax[0].scatter(X_spca[y==0, 0], X_spca[y==0, 1],
			  color='red', marker='^', alpha=0.5)
ax[0].scatter(X_spca[y==1, 0], X_spca[y==1, 1],
			  color='blue', marker='o', alpha=0.5)
ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')

# second figure ( we keep only one component)
# np.zeros = 500 because we initialize n_samples = 1000
# in make_circle function
ax[1].scatter(X_spca[y==0, 0], np.zeros((500,1))+0.02,
			  color='red', marker='^', alpha=0.5)
ax[1].scatter(X_spca[y==1, 0], np.zeros((500,1))-0.02,
			  color='blue', marker='o', alpha=0.5)
ax[1].set_ylim([-1, 1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC1')

plt.show()

# Using Kernel PCA with a good value of gamma :
from kernelPCA import rbf_kernel_pca

X_kpca, lambdas = rbf_kernel_pca(X, gamma=15, n_components=2)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14,6))

# first figure (it already changes, so we can separate afterwards
# using only one component)
ax[0].scatter(X_kpca[y==0, 0], X_kpca[y==0, 1],
			  color='red', marker='^', alpha=0.5)
ax[0].scatter(X_kpca[y==1, 0], X_kpca[y==1, 1],
			  color='blue', marker='o', alpha=0.5)
ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')


ax[1].scatter(X_kpca[y==0, 0], np.zeros((500,1))+0.02,
			  color='red', marker='^', alpha=0.5)
ax[1].scatter(X_kpca[y==1, 0], np.zeros((500,1))-0.02,
			  color='blue', marker='o', alpha=0.5)
ax[1].set_ylim([-1, 1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC1')

# Note the complex task reside in finding a good gamma value !!

plt.show()