from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=150,
				  n_features=2, # 2 dim
				  centers=3,
				  cluster_std=0.5, # variance
				  shuffle=True,
				  random_state=0)

import matplotlib.pyplot as plt

plt.scatter(X[:, 0],
			X[:, 1],
			c='white',
			marker='o',
			s=50)

plt.grid()
plt.show()


# k-means using KMeans class from scikit-learn's cluster module
from sklearn.cluster import KMeans

# initialize K-means with parameters :
# n_clusters : nb of desired clusters to 3 (specifying the
# nb of clusters a priori is one of the limitations of k-means)
# n_init=10 : to run k-means algo 10 times independently with
# different random centroids to choose the final model as the one
# with the lowest SSE.
# max_iter : max nb of iterations for each single run
# tol : stop the algorithm if the within-cluster SSE changes less than
# 1e-04 here.
km = KMeans(n_clusters=3,
			init='random', # init='k-means++' (default) will use k-means++
			n_init=10,
			max_iter=300,
			tol=1e-04,
			random_state=0)

y_km = km.fit_predict(X)


# visualize the clusters that k-means identified
plt.scatter(X[y_km==0,0], # abscisses of all pts belonging to cluster 1 (0)
			X[y_km==0,1], # ords of all pts belonging to cluster 1 (0)
			s=50, # thickness
			c='lightgreen',
			marker='s',
			label='cluster 1')

plt.scatter(X[y_km==1,0],
			X[y_km==1,1],
			s=50,
			c='orange',
			marker='o',
			label='cluster 2')

plt.scatter(X[y_km==2,0],
			X[y_km==2,1],
			s=50,
			c='lightblue',
			marker='v',
			label='cluster 3')

plt.scatter(km.cluster_centers_[:,0], # all centroids abscisses
			km.cluster_centers_[:,1], # all centroids ords
			s=250,
			marker='*',
			c='red',
			label='centroids')

plt.legend()
plt.grid()
plt.show()

print('Distorsion: %.2f' % km.inertia_)