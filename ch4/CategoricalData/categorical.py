# categorical data are divided into 2 groups 
#	- ordinal (like XL > L > M) for t-shirt size
#	- nominal like red, blue, green (color of t-shirt)

import pandas as pd

df = pd.DataFrame([
			['green', 'M', 10.1, 'class1'],
			['red', 'L', 13.5, 'class2'],
			['blue', 'XL', 15.3, 'class1']])

df.columns = ['color', 'size', 'price', 'classlabel']

# to make sure the learning algo interprets the ordinal features
# correctly, we need to convert the cagtegorical string values
# into integers. We have to define this mapping manually :
size_mapping = {
	'XL': 3,
	'L': 2,
	'M': 1
} # for example

df['size'] = df['size'].map(size_mapping) # map size col

print(df)

# to transform the integer values back to the original string
# we can simply define a reverse-mapping dict :
# inv_size_mapping = {v: k for k, v in size_mapping.items()}
# and use it with th emap methode from pandas.