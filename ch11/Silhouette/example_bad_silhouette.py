# we gonna apply silhouette on a only 2 clusters (even if there seem
# to have 3 clusters) to see silhouette on a bad clustering

from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=150,
				  n_features=2, # 2 dim
				  centers=3,
				  cluster_std=0.5, # variance
				  shuffle=True,
				  random_state=0)

from sklearn.cluster import KMeans

km = KMeans(n_clusters=2, init='k-means++', n_init=10,
			max_iter=300, tol=1e-04, random_state=0)

y_km = km.fit_predict(X)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.metrics import silhouette_samples

plt.scatter(X[y_km==0,0],
			X[y_km==0,1],
			s=50, c='lightgreen',
			marker='s',
			label='cluster 1')

plt.scatter(X[y_km==1,0],
			X[y_km==1,1],
			s=50,
			c='orange',
			marker='o',
			label='cluster 2')

plt.scatter(km.cluster_centers_[:,0],
			km.cluster_centers_[:,1],
			s=250,
			marker='*',
			c='red',
			label='centroids')

plt.legend()
plt.grid()
plt.show()


# create the silhouette plot to evaluate the results

# np.unique([1, 2, 5, 2, 3, 3, 1])
# return : [1, 2, 5, 3]
cluster_labels = np.unique(y_km) # keep unique number
n_clusters = cluster_labels.shape[0] # nb of cluster (nb of elem in the array or in the row : shape[0])
silhouette_vals = silhouette_samples(X, y_km, metric='euclidean')

y_ax_lower, y_ax_upper = 0, 0
yticks = []
for i, c in enumerate(cluster_labels):
	c_silhouette_vals = silhouette_vals[y_km==c] # get only silhouette from current cluster
	c_silhouette_vals.sort()
	y_ax_upper += len(c_silhouette_vals) # to place the bar
	color = cm.jet(i / n_clusters)
	plt.barh(range(y_ax_lower, y_ax_upper), # y coord(s) of the bar
			 c_silhouette_vals, # width of the bar
			 height=1.0,
			 edgecolor='none',
			 color=color)
	yticks.append((y_ax_lower + y_ax_upper) / 2)
	y_ax_lower += len(c_silhouette_vals)

# append the mean of all silhouette to the graph
silhouette_avg = np.mean(silhouette_vals)
plt.axvline(silhouette_avg, color='red', linestyle='--')

# plot labels in yticks (y coord) position
plt.yticks(yticks, cluster_labels + 1)
plt.ylabel('Cluster')
plt.xlabel('Silhouette coefficient')
plt.show()

# diffrent lengths and width ==> no very good clustering