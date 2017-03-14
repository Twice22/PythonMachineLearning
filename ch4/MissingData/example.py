# sometimes we are missing values for various reasons

import pandas as pd
from io import StringIO

csv_data = '''A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
0.0,11.0,12.0,'''

df = pd.read_csv(StringIO(csv_data))
print(df)
print(df.isnull().sum()) # number of missing values per col

print(df.dropna()) # remove row where there are missing values
print(df.dropna(axis=1)) # remove col where data are missing

# only drop rows where all columns are NaN
# >>> df.dropna(how='all')
# drop rows that have not at least 4 non-NaN values
# >>> df.dropna(thresh=4)
# only drop rows where NaN appear in specific columns (here: 'C')
# >>> df.dropna(subset=['C'])


# drawback : we can remove too many data and missing valuable info.
