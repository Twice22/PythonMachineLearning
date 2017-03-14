# we need to install wtforms via : pip install wtforms from cmd
# wtforms allow us to use forms in our web app

from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators

app = Flask(__name__)

# embeded textField in the page
class HelloForm(Form):
	sayhello = TextAreaField('', [validators.DataRequired()])

@app.route('/')
def index():
	form = HelloForm(request.form)
	return render_template('first_app.html', form=form)

@app.route('/hello', methods=['POST'])
def hello():
	form = HelloForm(request.form)
	if request.method == 'POST' and form.validate():
		name = request.form['sayhello']
		return render_template('hello.html', name=name)
	return render_template('first_app.html', form=form)

if __name__ == '__main__':
	app.run(debug=True)