import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(123)
variables = ['X', 'Y', 'Z']
labels = ['ID_0', 'ID_1', 'ID_2', 'ID_3', 'ID_4']

# matrix 5x3 with random float between 0 and 10
X = np.random.random_sample([5,3])*10

df = pd.DataFrame(X, columns=variables, index=labels)
# print(df)

# Step 1 : compute the distance matrix
# we use SciPy's spatial.distance submodule :
from scipy.spatial.distance import pdist, squareform

# pdist is the condensed distance matrix
# is used as input to squareform to create sym matrix
row_dist = pd.DataFrame(squareform(
				pdist(df, metric='euclidean')),
				columns=labels, index=labels)

# print(row_dist)

# we'll apply the complete linkage agglomeration using linkage
# fct from SciPy's cluster.hierarchy submodule.
# Note : linkage fct take a condensed distance matrix (pdist) as input
# and not for example the row_dist that we get from squreform fct
from scipy.cluster.hierarchy import linkage
row_clusters = linkage(pdist(df, metric='euclidean'), method='complete')

# row_clusters = linkage(df.values, method='complete', metric='euclidean') # works too

# to visualize the linkage matrix :
pd.DataFrame(row_clusters,
			 columns=['row label 1', # most dissimilar (distant)
			 		  'row label 2', # members of each cluster
			 		  'distance', # distance between those members
			 		  'no. of items in clust.'],
			 index=['cluster %d' %(i+1) for i in
			 		range(row_clusters.shape[0])])


# visualize the results in the form of a dendrogram :
from scipy.cluster.hierarchy import dendrogram
# make dendrogram black (part 1/2)
# from scipy.cluster.hierarchy import set_link_color_palette
# set_link_color_palette(['black'])

row_dendr = dendrogram(row_clusters, labels=labels
					   # make dendrogram black (part 2/2)
					   # color_threshold=np.inf
					   )

plt.tight_layout()
plt.ylabel('Euclidean distance')
plt.show()

# dendogram graph shows the diffrent clusters that were formed during
# the agglomerative hierarchical clustering (ID_0, ID_4 and ID_1, ID_2
# are the most similar ones based on the Euclidean distance metric).


# in pratical it's best to attach dendrograms to a heat map :

# Step 1
# define figure object and x and y axis position, width and height
# of the dendrogram using add_axes
# we rotate the dendrogram 90 degrees counter-clockwise.
fig = plt.figure(figsize=(8,8))
axd = fig.add_axes([0.09, 0.1, 0.2, 0.6])
row_dendr = dendrogram(row_clusters, orientation='right')

# Step 2 
# we reorder the data in our initial DataFrame according to the
# clustering labels that can be accessed from the dendrogram obj,
# which is a Python dict, via the leaves key
# ix : A primarily label-location based indexer, with integer position fallback.
df_rowclust = df.ix[row_dendr['leaves'][::-1]]

# Step 3
# we construct the heat map from the reordered DataFrame and position
# it right next to the dendrogram
axm = fig.add_axes([0.23, 0.1, 0.6, 0.6])
cax = axm.matshow(df_rowclust, interpolation='nearest', cmap='hot_r')

# Step 4
# modify the aesthetics

# remove axis ticks
axd.set_xticks([])
axd.set_yticks([])
# hide axis spines
for i in axd.spines.values():
	i.set_visible(False)
fig.colorbar(cax) # add color bar
# assign feature name to x axis tick labels
axm.set_xticklabels([''] + list(df_rowclust.columns))
# assign sample name to y axis tick labels
axm.set_yticklabels([''] + list(df_rowclust.index))
plt.show()


# Note : there is a scikit-learn implementation which allows
# us to choose the nb of clusters that we want to return (n_clusters)
from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters=2, affinity='euclidean',
							 linkage='complete')
labels = ac.fit_predict(X)
print('Cluster labels: %s' % labels)
# return [0 1 1 0 0]
# that means the first, fourth and fifth sample (ID_0, ID_3, ID_4)
# were assigned to one cluster (0)
# and ID_1, ID_2 to another one (1). It's consistent with what we
# had found earlier.