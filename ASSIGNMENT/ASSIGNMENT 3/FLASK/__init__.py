from flask import Flask, render_template, redirect, url_for, request
import pandas as pd
import os
import numpy as np
import pickle
app = Flask(__name__)
	
@app.route('/webform',methods=['POST','GET'])
def webform():
	return render_template('webform.html')

	#this route will get the input from forms, calculate prediction and pass it on to result.html page
@app.route('/output', methods=['POST','GET'])
def get_output(): 
	if request.method =='POST':
		data = {}
		form_data = request.form
		data['form'] = form_data
		predict_A = float(form_data['predict_Lever position'])
		predict_B = float(form_data['predict_Ship speed'])
		predict_C = float(form_data['predict_Gas Turbine shaft torque'])
		predict_D = float(form_data['predict_Gas Turbine rate of revolutions'])
		predict_E = float(form_data['predict_Gas Generator rate of revolutions '])
		predict_F = form_data['predict_Starboard Propeller Torque']
		predict_G = float(form_data['predict_Port Propeller Torque'])
		predict_H = float(form_data['predict_HP Turbine exit temperature'])
		predict_I = float(form_data['predict_GT Compressor inlet air temperature'])
		predict_J = float(form_data['predict_GT Compressor outlet air temperature'])
		predict_K = float(form_data['predict_(P48)'])
		predict_L = form_data['predict_(P1)']
		predict_M = float(form_data['predict_(P2)'])
		predict_N = float(form_data['predict_(Pexh)'])
		predict_O = form_data['predict_(TIC)']
		predict_P = float(form_data['predict_(mf)'])
		
		#if(predict_A !="" && predict_B!="" && predict_C!="" && predict_D!="" && predict_E!="" && predict_F!="" && predict_G!="" && predict_H!="" && predict_I !="" && predict_J!="" &&  predict_K !="" && predict_L!="" && predict_M!="" && predict_N!="" && predict_O!="" && predict_P!=""):
		
		input_data = np.array([predict_A, predict_B, predict_C, predict_D, predict_E, predict_F, predict_G, predict_H, predict_I, predict_J, predict_K, predict_L, predict_M, predict_N, predict_O, predict_P])
		columns = ['1 - Lever position (lp) [ ]', '2 - Ship speed (v) [knots]', 
                                          '3 - Gas Turbine shaft torque (GTT) [kN m]',
                                          '4 - Gas Turbine rate of revolutions (GTn) [rpm]', 
                                          '5 - Gas Generator rate of revolutions (GGn) [rpm]', 
                                          '6 - Starboard Propeller Torque (Ts) [kN]', '7 - Port Propeller Torque (Tp) [kN]', 
                                          '8 - HP Turbine exit temperature (T48) [C]', 
                                          '9 - GT Compressor inlet air temperature (T1) [C]',
                                          '10 - GT Compressor outlet air temperature (T2) [C]', 
                                          '11 - HP Turbine exit pressure (P48) [bar]', 
                                          '12 - GT Compressor inlet air pressure (P1) [bar]',
                                          '13 - GT Compressor outlet air pressure (P2) [bar]', 
                                          '14 - Gas Turbine exhaust gas pressure (Pexh) [bar]',
                                          '15 - Turbine Injecton Control (TIC) [%]', '16 - Fuel flow (mf) [kg/s]']

		df = pd.DataFrame(input_data.reshape(-1, len(input_data)),columns=columns)
				
			#get the target
	
		#my_dir = os.path.dirname('/var/www/FlaskApp/FlaskApp/')
		my_dir = os.getcwd()
		pickle_file_path = os.path.join(my_dir,'KNN.pkl')
		print(pickle_file_path)
		model= pickle.load(open("/var/www/FlaskApp/FlaskApp/KNN.pkl"),'rb')
		prediction = 12
		# = open('KNN.pkl', 'rb')
		
	#	prediction= pickle_file_path 
		#	prediction=model.predict(df)
		return render_template('predict.html',prediction =prediction)
	else:
		message ="Kindly enter all valid values"
		return redirect(url_for('webform', message=message))
	


@app.route('/validatelogin', methods=['POST','GET'])
def validatelogin():
	error = None
	if request.method == 'POST':
		if request.form['username'] != 'admin' or request.form['password'] != 'admin':
			error = 'Invalid Credentials. Please Try Again'
			return render_template('login.html', error = error) #will render login page again with an error displayed
		else:
			return redirect(url_for('webform')) # this will call webfrm route and will render webform.html
	
	
#@app.route('/login', methods = ['GET','POST'])
#this is my start of the application, it will load login.html page
@app.route('/') 
def login():
	return render_template('login.html')

if __name__ == '__main__':
	app.run(debug = True)
