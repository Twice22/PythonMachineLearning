# machine learning libraries require that class labels are
# encoded as integer values. Class being ordinal we can just
# replace class with whatever number using enumeration :

import numpy as np
import pandas as pd

df = pd.DataFrame([
			['green', 'M', 10.1, 'class1'],
			['red', 'L', 13.5, 'class2'],
			['blue', 'XL', 15.3, 'class1']])

df.columns = ['color', 'size', 'price', 'classlabel']

class_mapping = {label:idx for idx, label in enumerate(np.unique(df['classlabel']))}

print(class_mapping) # {'class1': 0, 'class2': 1}

# transform class labels into integers
df['classlabel'] = df['classlabel'].map(class_mapping)
print(df)

# to revert it
inv_class_mapping = {v:k for k, v in class_mapping.items()}
df['classlabel'] = df['classlabel'].map(inv_class_mapping)
print(df)


# there is a convenient LabelEncoder class in scikit-learn
# that allows us to do the same :
# from sklearn.preprocessing import LabelEncoder
# class_le = LabelEncoder()
# y = class_le.fit_transform(df['classlabel'].values)
# print(y) # array([0, 1, 0])

# fit_transform is a shortcut to calling fit and transform
# class_le.inverse_transform(y) # give back the classes