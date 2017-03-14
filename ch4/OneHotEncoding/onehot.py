import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


df = pd.DataFrame([
			['green', 'M', 10.1, 'class1'],
			['red', 'L', 13.5, 'class2'],
			['blue', 'XL', 15.3, 'class1']])

df.columns = ['color', 'size', 'price', 'classlabel']

size_mapping = {
	'M': 1,
	'L': 2,
	'XL': 3
}

# change colum size with number
df['size'] = df['size'].map(size_mapping)

# return matrice of the values of the column we choose
X = df[['color', 'size', 'price']].values

# to transform nominal values into integer
color_le = LabelEncoder()
# X[:, 0] : select all row/ first col (0)
# transform values from first col to integers using LabelEncoder()
X[:, 0] = color_le.fit_transform(X[:, 0])
print(X)

# problem here !! A learning algorithm will now assume green (1)
# is larger than blue (0) and red (2)  larger than blue.

# A workaround is to use one-hot encoding. The idea is to create
# a new dummy feature for each unique value of the nominal
# feature col. Here we can convert color into 3 new features :
# blue, green, red :

# from sklearn.preprocessing import OneHotEncoder
# ohe = OneHotEncoder(categorical_features=[0]) # we are using features from first col (0)
# ohe.fit_transform(X).toarray()

# we now have :
# array([[ 0. , 1. , 0. , 1. , 10.1],
# 		 [ 0. , 0. , 1. , 2. , 13.5],
# 		 [ 1. , 0. , 0. , 3. , 15.3]])

# Note : by default, OneHotEncoder return a sparse matrice
# useful when there are lots of 0 in the matrice. Here we transform
# the sparse matrice into regular Numpy array using toarray.
# we could have done : OneHotEncoder(..., sparse=False)

# Note : another usefull way is :
# pd.get_dummies(df[['price', 'color', 'size']])