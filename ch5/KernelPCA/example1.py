# separating half-moon shapes

from sklearn.datasets import make_moons
from kernelPCA import * # homemade file

import matplotlib.pyplot as plt

X, y = make_moons(n_samples=100, random_state=123)

plt.scatter(X[y==0, 0], X[y==0, 1], color='red', marker='^', alpha=0.5)
plt.scatter(X[y==1, 0], X[y==1, 1], color='blue', marker='o', alpha=0.5)
plt.show()

# What will the dataset lookslike if we project it using
# standard PCA (clearly the datasets is not linearly separable)

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
ax[1].scatter(X_spca[y==0, 0], np.zeros((50,1))+0.02,
			  color='red', marker='^', alpha=0.5)
ax[1].scatter(X_spca[y==1, 0], np.zeros((50,1))-0.02,
			  color='blue', marker='o', alpha=0.5)
ax[1].set_ylim([-1, 1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC1')


# Note : we slightly shift the triangles and circles upwards
# and downwards so we can better visualize

# clearly we cannot separate the datasets

plt.show()


# let's try our kernel PCA function

from matplotlib.ticker import FormatStrFormatter
from kernelPCA import rbf_kernel_pca

alphas, lambdas = rbf_kernel_pca(X, gamma=15, n_components=1)

# fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14,6))

# # first figure (it already changes, so we can separate afterwards
# # using only one component)
# ax[0].scatter(X_kpca[y==0, 0], X_kpca[y==0, 1],
# 			  color='red', marker='^', alpha=0.5)
# ax[0].scatter(X_kpca[y==1, 0], X_kpca[y==1, 1],
# 			  color='blue', marker='o', alpha=0.5)
# ax[0].set_xlabel('PC1')
# ax[0].set_ylabel('PC2')
# # set the format of the x axis to 1 decimal digit
# ax[0].xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))


# ax[1].scatter(X_kpca[y==0, 0], np.zeros((50,1))+0.02,
# 			  color='red', marker='^', alpha=0.5)
# ax[1].scatter(X_kpca[y==1, 0], np.zeros((50,1))-0.02,
# 			  color='blue', marker='o', alpha=0.5)
# ax[1].set_ylim([-1, 1])
# ax[1].set_yticks([])
# ax[1].set_xlabel('PC1')
# # set the format of the x axis to 1 decimal digit
# ax[1].xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))

# # Note the complex task reside in finding a good gamma value !!

# plt.show()



# Let's assume taht the 26th point form half-moon dataset is a
# new data point x'. Our task is to project it onto this new
# subspace :

x_new = X[25] # 26th pt of half-moon

# original projection of pt X[25]
x_proj = alphas[25]

# return : Σ ( a(i) * k(x',x(i)).T )
def project_x(x_new, X, gamma, alphas, lambdas):
	# np.array : cast to numpy array
	# np.sum([np.sum(( x_new - row)**2) for row in X] return :
	#		| k(x_new, x(1)) |
	#	K =	| k(x_new, x(2)) |
	#		|      ...  	 |
	#		| k(x_new, x(n)) |
	pair_dist = np.array([np.sum(( x_new - row)**2)
						 for row in X])
	# calculate exp of all elt in the input array
	k = np.exp(-gamma * pair_dist)
	return k.dot(alphas / lambdas)

x_reproj = project_x(x_new, X,
					 gamma=15, alphas=alphas, lambdas=lambdas)

# let's visualize the projection on the first component :
plt.scatter(alphas[y==0, 0], np.zeros((50)),
			color='red', marker='^', alpha=0.5)
plt.scatter(alphas[y==1, 0], np.zeros((50)),
			color='blue', marker='o', alpha=0.5)
plt.scatter(x_proj, 0, color='black', label='original proj of pt X[25]',
			marker='^', s=100)
plt.scatter(x_reproj, 0, color='green',
			label='remapped point X[25]',
			marker='x', s=500) # s = size of the pts
# the number of points in the legend for scatter plot (nuage de pts)
plt.legend(scatterpoints=1)
plt.show()