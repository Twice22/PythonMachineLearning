# we can use interpolation techniques to replace the NaN data
# by the mean of all the number from row/col for example.

import pandas as pd
from io import StringIO
from sklearn.preprocessing import Imputer


csv_data = '''A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
0.0,11.0,12.0,'''


df = pd.read_csv(StringIO(csv_data))

# replace NaN with the mean (strategy=mean) of the value of the row (axis=0)
# could have used strategy='most_frequent' (replace NaN with most frequent value of row/col)
imr = Imputer(missing_values='NaN', strategy='mean', axis=0) # initialize param
imr = imr.fit(df) # learn param from the training data

# change the data using the previous now known parameters
imputed_data = imr.transform(df.values) 

print(imputed_data)