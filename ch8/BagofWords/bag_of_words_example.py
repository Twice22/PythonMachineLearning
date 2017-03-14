# we can use the CountVectorizer class implemented in scikit-learn
# CountVectorizer class takes an array of text data (doc or 
# sentences) and construct the bag-of-words for us :

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
count = CountVectorizer()
docs = np.array([
		'The sun is shining',
		'The weather is sweet',
		'The sun is shining and the weather is sweet'
	])

# fit the data to the classifier and transform the
# sentences into sparse feature vectors
bag = count.fit_transform(docs)

print(count.vocabulary_)

# we see the vocabulary is stored in a Python dict, which maps
# the unique words that are mapped to integer indices.


# print the features vectors that we just created
# we see a matrix with the nb of occurence of each words
# (a words is aliased to a number)
print(bag.toarray())


# scikit_implement of tf-idf (see bagofwords.txt)
from sklearn.feature_extraction.text import TfidfTransformer
tfidf = TfidfTransformer()

np.set_printoptions(precision=2) # precision of float
print(tfidf.fit_transform(bag).toarray())

# go back to bagofwords.txt to explanation