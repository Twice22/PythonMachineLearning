# Feature scaling is a very important step. we can use :
# - min-max scalling : xnorm = (x - xmin)/(xmax - xmin) :
# - standardization : xstd = (x - mean)/sigma

# standardization is often prefered because it creates a normal
# distribution with 0 mean and deviation = 1.

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# to split the sample into training and testing dataset
from sklearn.cross_validation import train_test_split

df_wine = pd.read_csv('wine.data', header=None)
df_wine.columns = ['Class label', 'Alcohol',
				   'Malic acid', 'Ash',
				   'Alcalinity of ash', 'Magnesium',
				   'Total phenols', 'Flavanoids',
				   'Nonflavanoid phenols',
				   'Proanthocyanins',
				   'Color intensity', 'Hue',
				   'OD280/OD315 of diluted wines',
				   'Proline']

# 3 classes : 1, 2, 3 that corresponds to different types of grapes
print('Class labels', np.unique(df_wine['Class label']))
df_wine.head()

# .iloc is primarily INTEGER position based
# .loc is primarily LABEL based
# X => matrice without first column ; y => first col vector (classes)
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# normalized or MinMax scalling
mms = MinMaxScaler()
X_train_norm = mms.fit_transform(X_train)
X_test_norm = mms.transform(X_test)

# standardized scalling
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)
