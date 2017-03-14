# we gonna use the Housing dataset.
# the features of the 506 samples may be summarized as :
# • CRIM: This is the per capita crime rate by town
# • ZN: This is the proportion of residential land zoned for lots larger than
#   25,000 sq.ft.
# • INDUS: This is the proportion of non-retail business acres per town
# • CHAS: This is the Charles River dummy variable (this is equal to 1 if tract
#   bounds river; 0 otherwise)
# • NOX: This is the nitric oxides concentration (parts per 10 million)
# • RM: This is the average number of rooms per dwelling
# • AGE: This is the proportion of owner-occupied units built prior to 1940
# • DIS: This is the weighted distances to five Boston employment centers
# • RAD: This is the index of accessibility to radial highways
# • TAX: This is the full-value property-tax rate per $10,000
# • PTRATIO: This is the pupil-teacher ratio by town
# • B: This is calculated as 1000(Bk - 0.63)^2, where Bk is the proportion of
# people of African American descent by town
# • LSTAT: This is the percentage lower status of the population
# • MEDV: This is the median value of owner-occupied homes in $1000s

# we will focus on the MEDV (housing prices) as our target var (y)
# that we want to predict using the 13 others variables

import pandas as pd

# the separator between values here is not the traditional ';' but \s+
df = pd.read_csv('housing.data', header=None, sep='\s+')

df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM',
			  'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B',
			  'LSTAT', 'MEDV']

# display the first 5 lines to be sure the dataset is load
# print(df.head())

# prior to training, we can visualize the important characteristics
# of a dataset : the EDA (Exploratory Data Analysis).
# It can help to visually detect the presence of outliers, the
# distribution of the data, and the relationships between features.

# we will create a scatterplot matrix to visualize the pair-wise
# correlations between the different features using seaborn lib :
import matplotlib.pyplot as plt # seaborn use matplotlib.pyplot
import seaborn as sns # pip install seaborn

sns.set(style='whitegrid', context='notebook') # notebook -> quadrillage

# 5 * 5 plots : 25 plot (1 feature for each axis for each plot)
cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']
sns.pairplot(df[cols], size=2.5)
plt.show()

# importing seaborn modify the default aesthetics of matplotlib
# use : sns.reset_orig() to reset default matplotlib settings


# on the EAD we see :
# 	- linear relationship between RM and MEDV
# 	- MEDV seems to be normally distributed but contains outliers


# to quantify the linear relationship between the features, we
# will create a correlation matrix (it is closely related to the
# covariance matrix that we have seen in the section about
# PCA). see p282


# we use corrcoef numpy function on the 5 features and we'll
# display a heatmap using heatmap from seaborn

import numpy as np
print(df[cols].values)
# df[cols].values return the values in a matrix (2-dim array)
# without the header
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.5)

# cm is a regular dataset
hm = sns.heatmap(cm, cbar=True, # draw color bar ?
				 annot=True, square=True, # axes aspect same size (each cell is a square)
				 fmt='.2f', # formatting code when annot=True (annotation on)
				 annot_kws={'size': 15}, # dict of key/value mapping if annot=True
				 yticklabels=cols, # plot the label of cols
				 xticklabels=cols) # plot the label of cols

plt.show()


# we can see that there is a strong correlation between :
#	- LSTAT and MEDV (-0.74) : but not linear if we use the AED plot previously
#	- RM and MEDV (0.70) : it looks linear according to EAD plot

# we will use RM and MEDV to plot a linear regression