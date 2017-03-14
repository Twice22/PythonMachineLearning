# Here is our Flask code, we will use the review.sqlite database file
# to store the movie reviews that are being submitted to our web app.

from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators # allow us to create forms

import pickle  # to unpickle and set up our classification model
import sqlite3
import os
import numpy as np

# import HashingVectorizer from vectorizer.py (vect = Hashing...)
from vectorizer import vect # our vectorizer.py file

app = Flask(__name__)

##### Preparing the Classifier
cur_dir = os.path.dirname(__file__)

# unpickle the register classifier :
# open it in read and binary mode : 'rb' and save it as clf
clf = pickle.load(open(os.path.join(cur_dir,
				  'pkl_objects/classifier.pkl'), 'rb'))

# return the path to the database
db = os.path.join(cur_dir, 'reviews.sqlite')

# fct to predict the class label and display it's probability
def classify(document):
	label = {0:'negative', 1:'positive'} # class 0 -> neg, 1 -> pos
	X = vect.transform([document])
	y = clf.predict(X)[0] # will return 0 or 1 (the class)
	proba = np.max(clf.predict_proba(X))
	return label[y], proba

# update the classifier given document and and y (class label)
def train(document, y):
	X = vect.transform([document])
	clf.partial_fit(X, [y])

# store a submitted movie review in our SQLite db
def sqlite_entry(path, document, y):
	conn = sqlite3.connect(path)
	c = conn.cursor()
	c.execute("INSERT INTO review_db (review, sentiment, date)"\
		" VALUES (?, ?, DATETIME('now'))", (document, y))
	conn.commit() # commit change
	conn.close()


######## Flask part

app = Flask(__name__)

# class that instantiates a TextAreaField which will be rendered
# in the reviewform.html
class ReviewForm(Form):
	# the data is required and should at least contains 15 char
	moviereview = TextAreaField('', [validators.DataRequired(),
									 validators.length(min=15)])

# the route road will use index function
@app.route('/')
def index():
	# render template reviewform.html in templates with
	# parameter form being the TextAreaField
	form = ReviewForm(request.form)
	return render_template('reviewform.html', form=form)

@app.route('/results', methods=['POST'])
def results():
	form = ReviewForm(request.form)
	# if request post and our text fit the validators from ReviewForm
	if request.method == 'POST' and form.validate():
		# fetch the contents of the submitted web form
		review = request.form['moviereview']
		# use our classify fct to determine the class
		y, proba = classify(review)
		# pass 3 var to the render : content, prediction....
		return render_template('results.html',
							   content=review,
							   prediction=y,
							   probability=round(proba*100, 2))
	# else render same as index
	return render_template('reviewform.html', form=form)

# this route will be reach essentially if the user click
# on feedback_button
@app.route('/thanks', methods=['POST'])
def feedback():
	feedback = request.form['feedback_button'] # is prediction correct or not ?
	review = request.form['review'] # save review in review var
	prediction = request.form['prediction'] # save pred in prediction var

	inv_label = {'negative': 0, 'positive': 1}
	y = inv_label[prediction]

	# if feedback incorrect, change the class to the opposite
	# before saving in databse
	if feedback == 'Incorrect':
		y = int(not(y))

	# update the classifier
	train(review, y)

	# save to database
	sqlite_entry(db, review, y)

	# display thanks.html page
	return render_template('thanks.html')

# import our update.py script to update our classifier
# from the SQLite db every time we restart the web app

if __name__ == '__main__':
	app.run(debug=True) # remove in production
	update_model(filepath=db, model=clf, batch_size=10000)