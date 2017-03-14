import pandas as pd

df = pd.read_csv('./movie_data.csv', encoding='ISO-8859-1')

import re # module for regex

def tokenizer(text):
	return text.split()

# We can use it in Python via the nltk : pip install nltk in cmd
from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()

def tokenizer_porter(text):
	# see the free book for more info about nltk :
	# http://www.nltk.org/book/
	return [porter.stem(word) for word in text.split()]

# to remove html tags, put smiley at the end and text to lower
def preprocessor(text):
	text = re.sub('<[^>]*>', '', text)
	emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
	text = re.sub('[\W]+', ' ', text.lower()) +\
		   ' '.join(emoticons).replace('-', '')

	return text

# apply function on the review col of the dataframe
# apply is a method from pandas
df['review'] = df['review'].apply(preprocessor)

# divide the dataset into 25k of training and 25k for testing

# the first 25000
X_train = df.loc[:25000, 'review'].values
y_train = df.loc[:25000, 'sentiment'].values

# the last 25000
X_test = df.loc[25000:, 'review'].values
y_test = df.loc[25000:, 'review'].values

# we see a GridSearchCV object to find optimal
# set of parameters (hyperparameters) for our logistic regression
# we use a 5-fold stratified cross-validation.
# remember GridSearchCV performs a brute force on set of hyperparams
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
# nltk.download('stopwords')

# load an apply English stop-word set :
# stop-word are a list of word without much meaning : 'is', 'a', 'the'...
from nltk.corpus import stopwords
stop = stopwords.words('english')

# initialize tf-idf (see bag_of_words_example.py and txt for info)
# tf-idf allows us to give smaller weight to meaningless words
# TfidfVectorizer combine CountVectorizer and TfidfTransformer !
tfidf = TfidfVectorizer(strip_accents=None,
					   lowercase=False,
					   preprocessor=None)

# param_grid constitued by 2 param dictionaries
	# first dictionary :
	# 	- use TfidfVectorizer with default settings
	# 		° use_idf=True
	# 		° smooth_idf=True
	# 		° norm='l2'
	# 	to calculate the tf-idfs

	# second dictionary :
	# 	- use TfidfVectorizer with
	# 		° use_idf=False,
	# 		° smooth_idf=False
	# 		° norm=None
	# 	so we train a model base on raw term frequencies !
# we train both model with 2 different norm for regularization :
# l2 and l1 and we train using different value of C (reg param) :
param_grid = [{'vect__ngram_range': [(1,1)], # 1-gram or bag-of-words
			  'vect__stop_words': [stop, None],
			  'vect__tokenizer': [tokenizer,
			  					  tokenizer_porter],
			  'clf__penalty': ['l1', 'l2'],
			  'clf__C': [1.0, 10.0, 100.0]},
			  {'vect__ngram_range': [(1,1)],
			   'vect__stop_words': [stop, None],
			   'vect__tokenizer': [tokenizer,
			   					   tokenizer_porter],
			   'vect__use_idf': [False],
			   'vect__norm': [None],
			   'clf__penalty': ['l1', 'l2'],
			   'clf__C': [1.0, 10.0, 100.0]}
			  ]

# will first execute tf-idf then logistic regression
lr_tfidf = Pipeline([('vect', tfidf),
					 ('clf', LogisticRegression(random_state=0))])

# to see all param that we can change in our param_grid !
# print(tfidf.get_params())

gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid,
						   scoring='accuracy',
						   cv=5, # use 5-fold cross-validation
						   verbose=1,
						   n_jobs=-1) # use all core from cpu

gs_lr_tfidf.fit(X_train, y_train)

# best result without Porter stemming, no stop-word lib
# and tf-idfs in combination with lr that uses L2 and C=10
print('Best parameter set: %s ' % gs_lr_tfidf.best_params_)

print('CV Accuracy: %.3f' % gs_lr_tfidf.best_score_)
clf = gs_lr_tfidf.best_estimator_
print('Test Accuracy:  %.3f' % clf.score(X_test, y_test))

# our machine learning model can predict whether a movie
# review is positive or negative with 90 % accuracy