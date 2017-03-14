from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=150,
				  n_features=2, # 2 dim
				  centers=3,
				  cluster_std=0.5, # variance
				  shuffle=True,
				  random_state=0)

from sklearn.cluster import KMeans

km = KMeans(n_clusters=3, # 3 clusters a priori
			init='k-means++', # use k-means++ algo (default)
			n_init=10, # run the algo 10 times (take the better one)
			max_iter=300, # stop iteration after 300
			tol=1e-04, # or in within-in SSE change less than 1e-10 in one iteration
			random_state=0)

# apply kmeans to sample
y_km = km.fit_predict(X)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.metrics import silhouette_samples

# np.unique([1, 2, 5, 2, 3, 3, 1])
# return : [1, 2, 5, 3]
cluster_labels = np.unique(y_km) # keep unique values
n_clusters = cluster_labels.shape[0] # nb of cluster by row
# n_clusters return the number of cluster here : 3

# return silhouette coef for each sample
silhouette_vals = silhouette_samples(X, y_km, metric='euclidean')

y_ax_lower, y_ax_upper = 0, 0
yticks = []
for i, c in enumerate(cluster_labels): # 3 iterations
	c_silhouette_vals = silhouette_vals[y_km == c] # get only silhouette_vals from cluster c
	c_silhouette_vals.sort()
	y_ax_upper += len(c_silhouette_vals)
	color = cm.jet(i / n_clusters) # color of cluster
	plt.barh(range(y_ax_lower, y_ax_upper), # y coord(s) of the bar
				   c_silhouette_vals, # width of the bar
				   height=1.0,
				   edgecolor='none',
				   color=color)
	yticks.append((y_ax_lower + y_ax_upper) / 2)
	y_ax_lower += len(c_silhouette_vals)

# append the mean of all silhouette to the graph
silhouette_avg = np.mean(silhouette_vals)
plt.axvline(silhouette_avg,
			color='red',
			linestyle='--')

# plot labels in yticks (y coord) position
plt.yticks(yticks, cluster_labels + 1)
plt.ylabel('Cluster')
plt.xlabel('Silhouette coefficient')
plt.show()

# we can see that we are closer to 1. So the clustering 
# seems good enough !

