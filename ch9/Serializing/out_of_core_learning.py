# 50k is a large datasets, so if we want to use the algorithm online
# or even larger dataset it would be impossible if we don't have
# enough memory or whatsoever, that is why we gonna use out-of-core
# technique :

# we already see stochastic gradient descent in chapter 2 which
# is a optimization algo that updates the model's weights using
# one sample at a time.

# first we define a tokenizer function that clean the data :
import numpy as np
import re
from nltk.corpus import stopwords

stop = stopwords.words('english')

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

# fct that reads in and returns one document at a time
def stream_docs(path):
	with open(path, 'r') as csv:
		next(csv) # skip header
		for line in csv:
			# line[:-3] all the line except the 3 last char
			# line[-2] the number at the end
			text, label = line[:-3], int(line[-2])
			yield text, label # return the data


# define a get_minibatch that take a doc stream from stream_docs
# fct and return a particular nb of doc specified by the size param
def get_minibatch(doc_stream, size):
	docs, y = [], []
	try:
		for _ in range(size):
			text, label = next(doc_stream)
			docs.append(text)
			y.append(label)
	except StopIteration:
		return None, None
	return docs, y

# we cannot use CountVectorizer for out-of-core learning since
# it requires holding the complete voc in memory... Idem for
# TfidfVectorizer..
# will use HashingVectorizer (https://sites.google.com/site/murmurhash/)
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier

# initialize hashingVectorizer with 2^21 features (to avoid hash collision)
# and using our tokenizer fct
vect = HashingVectorizer(decode_error='ignore',
						 n_features=2**21,
						 preprocessor=None,
						 tokenizer=tokenizer)

# reinitialized logistic regression with Stocastic Gradient Descend
# (SGD)with loss='log' --> logistic regression. 'hinge' --> SVM
clf = SGDClassifier(loss='log', random_state=1, n_iter=1)
doc_stream = stream_docs(path='./movie_data.csv')

# start out of core learning
# we use the progress bar from pyprind
import pyprind
pbar = pyprind.ProgBar(45)
classes = np.array([0, 1])
for _ in range(45):
	# 45 minibatch of 1000 doc each
	X_train, y_train = get_minibatch(doc_stream, size=1000)
	if not X_train:
		break
	X_train = vect.transform(X_train)
	# fit linear model with Stochastic Gradient Descent
	clf.partial_fit(X_train, y_train, classes=classes)
	pbar.update()

# we completed the learning process, use the last 5k doc
# to evaluate the perf of our model:
X_test, y_test = get_minibatch(doc_stream, size=5000)
X_test = vect.transform(X_test)
print('Accuracy: %.3f' % clf.score(X_test, y_test))

# use the last 5.000 docs to update our model:
clf = clf.partial_fit(X_test, y_test)