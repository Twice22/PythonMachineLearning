# we have now prepared our movie review dataset. How to split
# the text into individual elements ? As we use bag-of-word (1-gram)
# we can split the cleaned document as its whitespace char :

def tokenizer(text):
	return text.split()

# example
print(tokenizer('runners like running and thus they run'))

# another more advanced technique is 'word stemming', which is
# the process of transforming a word into its root form that
# allows us to map related words to the same stem !!!

# Note : see Porter stemmer algorithm
# We can use it in Python via the nltk : pip install nltk in cmd
from nltk.stem.porter import PorterStemmer

porter = PorterStemmer()
def tokenizer_porter(text):
	# see the free book for more info about nltk :
	# http://www.nltk.org/book/
	return [porter.stem(word) for word in text.split()]

# test :
print(tokenizer_porter('runners like running and thus they run'))

# Note : don't name the python file tokenize.py !!
# It will be a conflict between nltk.tokenize and this file


# other better stemmer algorithms :
# 		- Snowball stemmer
# 		- Lancaster stemmer
# they are accessible thru nltk package ! Note, stemmer can create
# non-real word. To avoid this we can use a technique call
# lemmatization (but computationally expensive)


# stop word : there are another cat of word, the one that don't
# bring much info.. 'is' 'has' 'a'... So we can remove them...
# to do so we use a set of 127 English stop-words from nltk lib
# which can be obtained by callung nltk.download fct :
import nltk
nltk.download('stopwords')

# load an apply English stop-word set :
from nltk.corpus import stopwords
stop = stopwords.words('english')

print([w for w in 
	tokenizer_porter('a runner likes running and runs a lot')
	if w not in stop])