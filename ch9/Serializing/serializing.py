# training machine learning model can be computationally quite
# expensive. We don't want to train our model every time we
# close our Python interpreter.
# So we can use Python in-build pickle module to serialize and
# de-serialize Python object !

from out_of_core_learning import *
import pickle
import os


# return path which is a concatenation of the values passed
dest = os.path.join('movieclassifier', 'pkl_objects')
print(dest)
if not os.path.exists(dest):
	# create movieclassifier directory and and subdir pkl_objects in it
	os.makedirs(dest)


# first arg : obj to pickle
# 2nd arg : an open file object where Python will write
# we open the file in write and binary mode : 'wb'
# we use protocol=4 : latest and most efficient pickle protocol

# serialize stop-words vocabulary so we don't have to
# install the NLTK voc on our server
pickle.dump(stop,
		open(os.path.join(dest, 'stopwords.pkl'), 'wb'),
		protocol=4)

# serialize our logistic regression
pickle.dump(clf, 
		open(os.path.join(dest, 'classifier.pkl'), 'wb'),
		protocol=4)


# note there is a better way to serialize array using :
# https://pypi.python.org/pypi/joblib