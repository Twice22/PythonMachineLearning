# executing the following code will create a new SQLite
# database inside movieclassifier directory

import sqlite3
import os

conn = sqlite3.connect('reviews.sqlite')

# cursor allow us to traverse over the db entries
c = conn.cursor()

c.execute('CREATE TABLE review_db'\
			' (review TEXT, sentiment INTEGER, date TEXT)')

example1 = 'I love this movie'
c.execute("INSERT INTO review_db"\
			" (review, sentiment, date) VALUES"\
			" (?, ?, DATETIME('now'))", (example1, 1))

example2 = 'I disliked this movie'
c.execute("INSERT INTO review_db"\
			" (review, sentiment, date) VALUES"\
			" (?, ?, DATETIME('now'))", (example2, 0))

# save the change we made to the database
conn.commit()
conn.close()