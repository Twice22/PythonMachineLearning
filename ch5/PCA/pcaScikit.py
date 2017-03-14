import pandas as pd

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

from plottingDecisionRegions import * # homemade file

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

# initialize param of PCA and Logistic Regression
pca = PCA(n_components=2) # nb of components to keep
lr = LogisticRegression()

# fit the model with X and apply the dimensionality reduction on X
X_train_pca = pca.fit_transform(X_train_std)
# apply dimensionality reduction on X
X_test_pca = pca.transform(X_test_std)

# fit the Logistic regression according to the training data
lr.fit(X_train_pca, y_train)

plot_decision_regions(X_train_pca, y_train, classifier=lr)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(loc='lower left')
plt.show()


# plot decision regions of the lr on the transformed test dataset
# we already fit the lr with the training data, so it works on the
# test data !!!
plot_decision_regions(X_test_pca, y_test, classifier=lr)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(loc='lower left')
plt.show()


# if we only want to see the variance ratio of each components :
# initialize PCA with n_components set to None (keep all components) and use
# explained_variance_ratio

# pca = PCA(n_componenets=None)
# X_train_pca = pca.fit_transform(X_train_std)
# pca.explained_variance_ratio_