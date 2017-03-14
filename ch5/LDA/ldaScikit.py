import pandas as pd

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

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

# keep 2 components
lda = LinearDiscriminantAnalysis(n_components=2)

# fit the model with X and apply the dimensionality reduction on X
X_train_lda = lda.fit_transform(X_train_std, y_train)

# Initialize Logistic Regression
lr = LogisticRegression()
lr = lr.fit(X_train_lda, y_train)

plot_decision_regions(X_train_lda, y_train, classifier=lr)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')

plt.show()

# we could lower the regularization strength to get a better result

# let's see how the test set is fit :
X_test_lda = lda.transform(X_test_std)
plot_decision_regions(X_test_lda, y_test, classifier=lr)
plt.ylabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.show()