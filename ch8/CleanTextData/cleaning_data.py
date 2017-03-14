# before building our bag-of-words model, we need to clean the
# text data by stripping ut of all unwanted character.

# to see why it's important, let's see the last 50 characters
# from the first document :

import pandas as pd

df = pd.read_csv('./movie_data.csv', encoding='ISO-8859-1')

# read first line (except header) : 0
# of the 'review' col and the last 50 character
# print(df.loc[0, 'review'][-50:])


# we see there are HTML markeup tags in the text. which is not
# a very usefull information. So we gonna get rid of those tags.
# For simplicity we gonna get rid of the punctuation too, but
# we'll keep the smiley... We use regex from re :
import re

# tutorial for reg expr :
# https://developers.google.com/edu/python/regular-expressions

def preprocessor(text):
	# re.sub(pattern, repl, string, count=0, flags=0)
	# Return the string obtained by replacing the leftmost
	# non-overlapping occurrences of pattern in string by
	# the replacement repl.


	# regex beginning with < not directly followed by > and ended by >
	# regex to remove all html tags
	text = re.sub('<[^>]*>', '', text)

	# (?:pattern) : Matches pattern but does not save the match; 
	# that is, the match is not stored for possible later use
	# ex : industr(?:y|ies) is equivalent to industry|industries.
	emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
	
	# remove all non word characters from the text via [\W]+
	# convert text into lowercase char, and ad eventually stored
	# emoticons to the end of the line (additionally we removed
	# the nose '-' for consistency)
	text = re.sub('[\W]+', ' ', text.lower()) +\
		   ' '.join(emoticons).replace('-', '')

	return text

# confirming our preprocessor work :
# print(preprocessor(df.loc[0, 'review'][-50:]))
# print(preprocessor("</a>This :) is :( a test :-)!"))

# let's apply our preprocessor function to all movie reviews in
# our DataFrame:
# apply is a method of the panda DataFrame
df['review'] = df['review'].apply(preprocessor)
