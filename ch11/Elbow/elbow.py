import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=150,
				  n_features=2, # 2 dim
				  centers=3,
				  cluster_std=0.5, # variance
				  shuffle=True,
				  random_state=0)

distortions = []

for i in range(1, 11): #1,2,...,10
	km = KMeans(n_clusters=i, # nb of desired cluster
				init='k-means++', # default value
				n_init=10, # run 10 times k-means algo
				max_iter=300,
				random_state=0)

	km.fit(X)

	distortions.append(km.inertia_)

plt.plot(range(1,11), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()

# we see k = 3 is a good choice on the graph