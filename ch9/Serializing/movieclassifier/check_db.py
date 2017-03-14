# file to check if the entries has been stored in our db
import sqlite3
import os

conn = sqlite3.connect('reviews.sqlite')
c = conn.cursor()
c.execute("SELECT * FROM review_db WHERE date"\
 		   " BETWEEN '2016-08-08 00:00:00' AND DATETIME('now')")

results = c.fetchall()
conn.close()
print(results)

# we can also use the Firefox plugin SQLite Manager