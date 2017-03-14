from sklearn.decomposition import KernelPCA
from sklearn.datasets import make_moons

import matplotlib.pyplot as plt

# create moon datasets (graph)
# it creates 2 moons with class label = 0 and 1 register
# in the vector y
X, y = make_moons(n_samples=100, random_state=123)

# initialize kernel parameters
scikit_kpca = KernelPCA(n_components=2,
						kernel='rbf', gamma=15)

# fit the model with X and apply the dimensionality reduction on X
X_skernpca = scikit_kpca.fit_transform(X)

# plot all x-axis based of class 0 (y==0) fct of the y-axis
plt.scatter(X_skernpca[y==0, 0], X_skernpca[y==0, 1],
			color='red', marker='^', alpha=0.5)
# plot all x-axis based of class 1 (y==1) fct of the y-axis
plt.scatter(X_skernpca[y==1, 0], X_skernpca[y==1, 1],
			color='blue', marker='o', alpha=0.5)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()