# we install PyPrind (Python Progress Indicator) using :
# pip install pyprind
# PyPrind allows us to see a progressbar to estimate how long it
# will take before completion

import pyprind
import pandas as pd
import os

pbar = pyprind.ProgBar(50000)
# a film is classified negative if he has less than 5/10
# a film is classified positive if he has more than 6/10
labels = {'pos':1, 'neg':0}

df = pd.DataFrame()
for s in ('test', 'train'):
	for l in ('pos', 'neg'):
		path='./aclImdb/%s/%s' %(s,l)
		# os.listdir(path) return a list containing the names
		# of the entries in the directory given by path
		for file in os.listdir(path):
			# open each file and read the content
			with open(os.path.join(path, file), 'r', encoding='utf-8') as infile:
				txt = infile.read()

			# DataFrame.append(other, ignore_index=False, verify_integrity=False)
			# other : DataFrame or Series/dict-like obj, or list of these
			#		  to append to the datz
			# ignore_index : If True, do not use the index labels. That
			#			     means it will append data and increment the
			#				 index for each newly created row
			df = df.append([[txt, labels[l]]], ignore_index=True)
			pbar.update()
df.columns = ['review', 'sentiment']

# since the class labels in the assembled dataset are sorted, we
# will now shuffle DataFrame using the permutation fct from the
# np.random submodule and store the file as CSV :
import numpy as np

np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))
df.to_csv('./movie_data.csv', index=False)

# reading the csv and printing an excerpt of the first 3
# samples to conform we have the right dataset format :
# df = pd.read_csv('./movie_data.csv')
# df.head(3)
