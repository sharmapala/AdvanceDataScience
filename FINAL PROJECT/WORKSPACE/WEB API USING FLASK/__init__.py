from flask import Flask, render_template, redirect, url_for, request, json
import pandas as pd
import os
import numpy as np
import pickle
import boto
from boto.s3.key import Key
import boto3
from flask import send_from_directory
import argparse
from flask import flash
from flask_mysqldb import MySQL
from flask import Markup
import urllib
from urllib.request import Request, urlopen

#place holder for module
app = Flask(__name__)

	
app.config['SECRET_KEY'] = 'secret'
app.config['MYSQL_HOST'] ='sql3.freemysqlhosting.net'
app.config['MYSQL_USER'] ='sql3234066'
app.config['MYSQL_PASSWORD'] = 'FvEy2x3RTS'
app.config['MYSQL_DB'] = 'sql3234066'
mysql = MySQL(app)

# def users():
#     cur = mysql.connection.cursor()
#     cur.execute('''SELECT user, host FROM mysql.user''')
#     rv = cur.fetchall()
#     return str(rv)

@app.route('/')
def home():
	cur = mysql.connection.cursor()
	cur.execute('''SELECT * FROM uniqCarrier''')
	uniqC = cur.fetchall()
	cur.execute('''SELECT * FROM Origin''')
	origin = cur.fetchall()
	cur.execute('''SELECT * FROM Destination''')
	dest = cur.fetchall()
	return render_template('home.html') 


@app.route('/showSignUp' , methods=['GET', 'POST'])
def showSignUp():
    return render_template('register.html')

@app.route('/signUp',methods=['POST', 'GET'])
def signUp():
    # create user code will be here !!
     # read the posted values from the UI
    if request.method == "POST":
     	_name = request.form['inputName']
     	_email = request.form['inputEmail']
     	_password = request.form['inputPassword']

     	if _name and _email and _password:
     		message = Markup("<h1>Successfully signUp</h1>")
     		flash(message)
     		return render_template('result.html')
     	else:
     		message = Markup("<h1>Fill all the fields</h1>")
     		flash(message)
     		return render_template('result.html')
    else:
    	return render_template('register.html')

@app.route('/predict', methods=['POST','GET'])
def predict(): 
	if request.method =='POST':
		data = {}
		form_data = request.form
		data['form'] = form_data
		year  = int(form_data['year'])
		month = int(form_data['month'])
		day = int(form_data['day'])
		dayofweek = int(form_data['dayofweek'])
		dest = form_data.get('dest')
		origin = form_data.get('origin')
		uniqC = form_data.get('unique_carrier')

		input_data = np.array([month,day, dayofweek, uniqC,origin,dest])

		columns = ['Month','DayofMonth','DayOfWeek','uniquecarrier_int','origin_int','dest_int']

         # creating feature dataframe
		feature_df = pd.DataFrame(input_data.reshape(-1, len(input_data)),columns=columns)

		#retrive pickle files from S3
		# req = Request("https://s3.amazonaws.com/dhanisha/delay.pkl")
		# delay_pkl = urlopen(req).read()
		print("url not working")
		urllib.request.urlretrieve("https://s3.amazonaws.com/dhanisha/delay.pkl", filename= 'delay.pkl')
		urllib.request.urlretrieve("https://s3.amazonaws.com/dhanisha/delay_type.pkl", filename= 'delay_type.pkl')
		urllib.request.urlretrieve("https://s3.amazonaws.com/dhanisha/delay_value.pkl", filename= 'delay_value.pkl')

		model1 = pickle.load(open('delay.pkl','rb'))
		#model2= pickle.load(open('delay_type.pkl','rb'))
		#model3= pickle.load(open('delay_value.pkl','rb'))
			
		predict_delay = model1.predict(feature_df)
		#predict_delay_type = model2.predict(feature_df)
		#predict_delay_value = model3.predict(feature_df)
		if predict_delay == 0:
			predict_delay = "No delay"
		else:
			predict_delay= "Delay"
		return render_template('predict.html', predict= predict_delay)
	else:
		return render_template('form.html', message="Kindly fill all the fields")




@app.route('/validatelogin', methods=['POST','GET'])
def validatelogin():
	error = None
	if request.method == 'POST':
		if request.form['username'] != 'admin' or request.form['password'] != 'admin':
			error = 'Invalid Credentials. Please Try Again'
			return render_template('login.html', error = error) #will render login page again with an error displayed
		else:
			cur = mysql.connection.cursor()
			cur.execute('''SELECT * FROM uniqCarrier''')
			uniqC = cur.fetchall()
			cur.execute('''SELECT * FROM Origin''')
			origin = cur.fetchall()
			cur.execute('''SELECT * FROM Destination''')
			dest = cur.fetchall()
			return render_template('form.html', uniqC = uniqC,  origin = origin,  dest = dest) # this will call webfrm route and will render webform.html

##This is my start of the application, it will load login.html page
@app.route('/login' , methods=['GET', 'POST']) 
def login():
	return render_template('login.html')

if __name__ == '__main__':
	app.run(debug = True)
