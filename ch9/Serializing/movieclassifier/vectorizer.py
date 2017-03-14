# we don't need to pickle HashingVectorizer in serializing.py
# since it does not need to be fitted.

from sklearn.feature_extraction.text import HashingVectorizer
import re
import os
import pickle

cur_dir = os.path.dirname(__file__)

# load the binary file stopword.pkl situated in pkl_projects folder
# in read and binary mode : 'rb'
stop = pickle.load(open(
					os.path.join(cur_dir, 'pkl_objects',
					'stopwords.pkl'), 'rb'))

def tokenizer(text):
	# remove html tag from text
	text = re.sub('<[^>]*>', '', text)

	# find all emoticons
	emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',
							text.lower())

	# replace non text [\W]+ by a space in (text lower + emoticon at the end)
	text = re.sub('[\W]+', ' ', text.lower()) \
			+ ' '.join(emoticons).replace('-', '')

	tokenized = [w for w in text.split() if w not in stop]

	# return each word from data that are not stop-words in an array
	return tokenized

# we use tokenizer that sendback data that are text + smileys at the end
vect = HashingVectorizer(decode_error='ignore',
						 n_features=2**21, # 2^21 to avoid collision
						 preprocessor=None,
						 tokenizer=tokenizer)

# load the binary file classifier.pkl in pkl_objects in rb mode
clf = pickle.load(open(
					os.path.join(cur_dir, 'pkl_objects',
					'classifier.pkl'), 'rb'))

import numpy as np
label = {0:'negative', 1:'positive'}
example = ['I love this movie']

# use HashingVectorizer to transform the simple example
# document into a word vector X
X = vect.transform(example)
print('Prediction: %s\nProbability: %.2f%%' 
	% (label[clf.predict(X)[0]], clf.predict_proba(X).max()*100))