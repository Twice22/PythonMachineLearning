# to install Flask : pip install flask in command-line
# the file contain the main code to run the Flask web application
# templates dir is the dir where Flask will loke for static HTML
from flask import Flask, render_template

# initialize a new Flask instance to let Flask know that it
# and can find HTML template folder (templates) in same dir
app = Flask(__name__)

# url that trigger the execution of the index function
@app.route('/')

def index():
	# render firs_app.html from templates dir
	return render_template('first_app.html')

if __name__ == '__main__':
	app.run()