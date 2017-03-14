# this file allow us to update the clf obj locally on our computer
# (having download the db with new entries) and we just have to
# upload the new pickle file.

# Indeed if we apply updates each time an user send feedback it
# would be computationally expensive...

import pickle
import sqlite3
import numpy as np
import os

# import HashingVectorizer (vect=HashingVectorizer in vectorizer.py)
from vectorizer import vect

def update_model(db_path, model, batch_size=10000):
	conn = sqlite3.connect(db_path)
	c = conn.cursor()
	c.execute('SELECT * from review_db')

	results = c.fetchmany(batch_size) # fetch batch of 10 000 samples
	while results:
		data = np.array(results)
		X = data[:, 0] # retrieve all row from col 0 : review
		y = data[:, 1].astype(int) # retrieve all row from col 1 : class

		# astype cast value of array to specific type

		classes = np.array([0, 1])
		X_train = vect.transform(X)
		clf.partial_fit(X_train, y, classes=classes)
		results = c.fetchmany(batch_size)

	conn.close()
	return None

cur_dir = os.path.dirname(__file__)

clf = pickle.load(open(os.path.join(cur_dir,
					   'pkl_objects',
					   'classifier.pkl'), 'rb'))
db = os.path.join(cur_dir, 'reviews.sqlite')

update_model(db_path=db, model=clf, batch_size=10000)

# Uncomment the following lines if you are sure that
# you want to update your classifier.pkl file permanently

# pickle.dump(clf, open(os.path.join(cur_dir,
#					    'pkl_objects', 'classifier.pkl'), 'wb')
#						, protocol=4)