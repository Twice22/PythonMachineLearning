import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=200, #100 pts for each moons
				  noise=0.05,
				  random_state=0)

plt.scatter(X[:, 0], X[:, 1])
plt.show()


# we first use k-means and complete linkage clustering
# to see if those previously discussed clustering algo can identify
# the half-moon shapes as separate clusters
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans

# f is the figure
# ax1 and ax2 are the sub plot
# we could have use f, ax = and then ax[0].scatter ax[1].scatter...
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,6)) # 1 row, 2 col

# initialize KMeans with parameters (n_cluster a priori)
km = KMeans(n_clusters=2,
			random_state=0)

# train KMeans
y_km = km.fit_predict(X)
ax1.scatter(X[y_km==0, 0],
			X[y_km==0, 1],
			c='lightblue',
			marker='o',
			s=40,
			label='cluster 1')
ax1.scatter(X[y_km==1, 0],
			X[y_km==1, 1],
			c='red',
			marker='s',
			s=40,
			label='cluster 2')

ax1.set_title('K-means clustering') # title of first graph

# initialize AgglomerativeClustering. We want 2 clusters
ac = AgglomerativeClustering(n_clusters=2,
							 affinity='euclidean',
							 linkage='complete')

# train AgglomerativeClustering
y_ac = ac.fit_predict(X)
ax2.scatter(X[y_ac==0,0],
			X[y_ac==0,1],
			c='lightblue',
			marker='o',
			s=40,
			label='cluster 1')
ax2.scatter(X[y_ac==1,0],
			X[y_ac==1,1],
			c='red',
			marker='s',
			s=40,
			label='cluster 2')

ax2.set_title('Agglomerative clustering')
plt.legend()
plt.show()

# that didn't work out well

# let's try dbscan
from sklearn.cluster import DBSCAN

# see dbscan.txt
db = DBSCAN(eps=0.2, # radius for core points
			min_samples=5, # nb of pts around to consider core pt
			metric='euclidean')

y_db = db.fit_predict(X)

plt.scatter(X[y_db==0,0],
			X[y_db==0,1],
			c='lightblue',
			marker='o',
			s=40,
			label='cluster 1')
plt.scatter(X[y_db==1,0],
			X[y_db==1,1],
			c='red',
			marker='s',
			s=40,
			label='cluster 2')
plt.legend()
plt.show()