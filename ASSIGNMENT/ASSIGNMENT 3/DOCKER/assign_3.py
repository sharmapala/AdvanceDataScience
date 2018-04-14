
# coding: utf-8

# In[1]:


import zipfile
import csv 
import pandas as pd
import sklearn
from sklearn.preprocessing import MinMaxScaler
import os
import numpy as np
import urllib
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.feature_selection import RFE, f_regression
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, RandomizedLasso)
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import (RandomForestRegressor , RandomForestClassifier , ExtraTreesRegressor , ExtraTreesClassifier)
import numpy as np
import pickle
from sklearn.metrics import (mean_squared_error,mean_absolute_error,r2_score , accuracy_score ,classification_report,silhouette_score) 
from sklearn.cross_validation import train_test_split
from sklearn import neural_network
from sklearn.neural_network import MLPRegressor ,MLPClassifier
from sklearn import svm 
from sklearn.svm import SVC ,SVR
from sklearn.neighbors import KNeighborsClassifier , KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
import boto3
import sys
import datetime
from sklearn.pipeline import Pipeline ,make_pipeline
from sklearn.preprocessing import LabelEncoder
import plotly.plotly as py
import plotly.graph_objs as go
from sklearn.multioutput import MultiOutputRegressor
import boto
from boto.s3.key import Key
import argparse


# In[ ]:


parser = argparse.ArgumentParser(description='Please enter the values below :')
parser.add_argument("--Access_key", help = 'Enter Access_key of your aws account')
parser.add_argument("--Secret_key", help = 'Enter Secret_key of your aws account')
args = parser.parse_args()

Access_key = args.Access_key
Secret_key = args.Secret_key

#if not os.path.exists(r'C:\Program Files\Docker Toolbox\ASSIGNMENT_3_PROPULSIONS'):
#    os.makedirs(r'C:\Program Files\Docker Toolbox\ASSIGNMENT_3_PROPULSIONS')
    
#zip = zipfile.ZipFile(r'C:\Program Files\Docker Toolbox\ASSIGNMENT_3_PROPULSIONS\UCI+CBM+Dataset')  
#zip.extractall(r'C:\Program Files\Docker Toolbox\ASSIGNMENT_3_PROPULSIONS')

df = r'/usr/src/assign3/data.txt'

f = open(df , 'r')
my_data = f.readlines() #converting into list
my_arr = np.array(my_data) # making it into numpy array

arr = []
for k in range(11934):
    i = 3
    while(i<289):
        temp = float(my_arr[k][i:i+13])
        arr.append(temp)
        i = i+16


arr = np.array(arr)
arr = arr.reshape(11934,18)

my_final_data = pd.DataFrame(arr , columns = ['1 - Lever position (lp) [ ]', '2 - Ship speed (v) [knots]', 
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
                                              '15 - Turbine Injecton Control (TIC) [%]', '16 - Fuel flow (mf) [kg/s]', 
                                              '17 - GT Compressor decay state coefficient.', 
                                              '18 - GT Turbine decay state coefficient.'])
my_final_data.to_csv('data.csv')
    #return data
    #return my_final_data

                                        ### DATA CELANING ###
    

df = pd.read_csv('data.csv')
num_of_missing_values = df.isnull().sum().sum()
if ( num_of_missing_values == 0) :
    final_file = df
    #final_file.to_csv('final_file.csv')
else :
    final_file = df.fillna(method='bfill' ,axis = 0).fillna('0')

#final_1.append(final_file)
final_file = final_file.to_csv('clean_data.csv')
    #return final_file

#cleaning(r'C:\ASSIGNMENT_3_PROPULSIONS\UCI CBM Dataset\final_file.csv')
'''
buck_name="dhanisha" #enter bucket name
Input_location = 'us-east-1'
S3_client = boto3.client('s3', Input_location, aws_access_key_id= Access_key, aws_secret_access_key= Secret_key)

if Input_location == 'us-east-1':
        S3_client.create_bucket(
            Bucket=buck_name
        )
else:
    S3_client.create_bucket(
            Bucket=buck_name,
            CreateBucketConfiguration={'LocationConstraint': Input_location},
        )

S3_client.upload_file('clean_data.csv', buck_name, 'clean_data.csv')
'''

#data = urllib.request.urlopen('clean_data.csv')
#srcFileName="clean_data.csv"
#bucketName="dhanisha"
#obj = S3_client.get_object(Bucket="dhanisha", Key = srcFileName)
#clean = pd.read_csv(obj['Body'])
#clean = pd.read_csv(data)



srcFileName="clean_data.csv" # filename on S3
#destFileName="s3_abc.txt" # output file name
bucketName="dhanisha" # S3 bucket name 

conn = boto.connect_s3(Access_key,Secret_key)
bucket = conn.get_bucket(bucketName)

#Get the Key object of the given key, in the bucket
k = Key(bucket,srcFileName)

#Get the contents of the key into a file 
k.get_contents_to_filename(srcFileName)
clean = pd.read_csv(k)
clean.to_csv('clean.csv')

                                        ### DATA CELANING ###
    
def cleaning(clean):
    df = pd.read_csv(clean)
    num_of_missing_values = df.isnull().sum().sum()
    if ( num_of_missing_values == 0) :
        final_file = df
        #final_file.to_csv('final_file.csv')
    else :
        final_file = df.fillna(method='bfill' ,axis = 0).fillna('0')
        
    #final_1.append(final_file)
    final_file = final_file.to_csv('clean_data1.csv')
    return final_file
    
#cleaning(r'C:\ASSIGNMENT_3_PROPULSIONS\UCI CBM Dataset\final_file.csv')

                                        ### EXPLORATORY DATA ANALYSIS ###

def eda(eda):
    df = pd.read_csv(eda)
    
    ### correlation plot ###
    corr = df.corr() 
    plt.figure(figsize=(10, 10))

    sns.heatmap(corr, 
            cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.5,
            annot=True, annot_kws={"size": 8}, square=True)
    #plt.show()
    
    ### Box plots to find the relation between variables ###
    
'''
    trace0 =[ go.Bar(
            x=df['18 - GT Turbine decay state coefficient.'],
            y=df['3 - Gas Turbine shaft torque (GTT) [kN m]']
             
        )]
    
    layout = go.Layout(
        xaxis =dict(
            title = 'TURBINE DECAY RATE'
        ),
        yaxis = dict(
            title = 'TURBINE SHAFT TORQUE'
        )
    )
    labels = [trace0]
    py.iplot(labels, filename=trace0)
    

    trace1 =[ go.Bar(
            x=df['18 - GT Turbine decay state coefficient.'],
            y=df['4 - Gas Turbine rate of revolutions (GTn) [rpm]']
        )]
    layout = go.Layout(
        xaxis =dict(
            title = 'TURBINE DECAY RATE'
        ),
        yaxis = dict(
            title = 'TURBINE RATE OF REVOLUTIONS'
        )
    )
    labels = [trace1]
    py.iplot(labels, filename=trace01)
    
    trace2 = [go.Bar(
            x=df['18 - GT Turbine decay state coefficient.'],
            y=df['5 - Gas Generator rate of revolutions (GGn) [rpm]']
    )]
    layout = go.Layout(
        xaxis =dict(
            title = 'TURBINE DECAY RATE'
        ),
        yaxis = dict(
            title = 'GENERATOR RATE OF REVOLUTIONS'
        )
    )
    labels = [trace2]
    py.iplot(labels, filename=trace02)
    
    trace3 = [go.Bar(
            x=df['18 - GT Turbine decay state coefficient.'],
            y=df['6 - Starboard Propeller Torque (Ts) [kN]']
    )]
    layout = go.Layout(
        xaxis =dict(
            title = 'TURBINE DECAY RATE'
        ),
        yaxis = dict(
            title = 'STARBOARD PROPELLER TORQUE'
        )
    )
    labels = [trace3]
    py.iplot(labels, filename=trace03)
    
    trace4 = [go.Bar(
             x=df['18 - GT Turbine decay state coefficient.'],
             y=df['7 - Port Propeller Torque (Tp) [kN]']
    )]
    layout = go.Layout(
        xaxis =dict(
            title = 'TURBINE DECAY RATE'
        ),
        yaxis = dict(
            title = 'PORT PROPELLER TORQUE'
        )
    )
    labels = [trace4]
    py.iplot(labels, filename=trace04)
    
    trace5 = [go.Bar(
            x=df['18 - GT Turbine decay state coefficient.'],
            y=df['8 - HP Turbine exit temperature (T48) [C]']
    )]
    layout = go.Layout(
        xaxis =dict(
            title = 'TURBINE DECAY RATE'
        ),
        yaxis = dict(
            title = 'TURBINE EXIT TEMPERATURE'
        )
    )
    labels = [trace5]
    py.iplot(labels, filename=trace05)
    
    trace6 = [go.Bar(
            x=df['18 - GT Turbine decay state coefficient.'],
            y=df['9 - GT Compressor inlet air temperature (T1) [C]']
    )]
    layout = go.Layout(
        xaxis =dict(
            title = 'TURBINE DECAY RATE'
        ),
        yaxis = dict(
            title = 'COMPRESSOR INLET AIR TEMPERATURE'
        )
    )
    labels = [trace6]
    py.iplot(labels, filename=trace06)
    
    trace7 = [go.Bar(
            x=df['18 - GT Turbine decay state coefficient.'],
            y=df['12 - GT Compressor inlet air pressure (P1) [bar]']
    )]
    layout = go.Layout(
        xaxis =dict(
            title = 'TURBINE DECAY RATE'
        ),
        yaxis = dict(
            title = 'COMPRESSOR INLET AIR PRESSURE'
        )
    )
    labels = [trace7]
    py.iplot(labels, filename=trace07)
    
    trace8 = [go.Bar(
            x=df['18 - GT Turbine decay state coefficient.'],
            y=df['13 - GT Compressor outlet air pressure (P2) [bar]']
    )]
    layout = go.Layout(
        xaxis =dict(
            title = 'TURBINE DECAY RATE'
        ),
        yaxis = dict(
            title = 'COMPRESSOR OUTLET AIR PRESSURE'
        )
    )
    labels = [trace8]
    py.iplot(labels, filename=trace08)
    
    trace9 = [go.Bar(
            x=df['18 - GT Turbine decay state coefficient.'],
            y=df['14 - Gas Turbine exhaust gas pressure (Pexh) [bar]']
    )]
    layout = go.Layout(
        xaxis =dict(
            title = 'TURBINE DECAY RATE'
        ),
        yaxis = dict(
            title = 'TURBINE EXHAUST GAS PRESSURE'
        )
    )
    labels = [trace9]
    py.iplot(labels, filename=trace09)
    
    trace10 = [go.Bar(
            x=df['18 - GT Turbine decay state coefficient.'],
            y=df['15 - Turbine Injecton Control (TIC) [%]']
    )]
    layout = go.Layout(
        xaxis =dict(
            title = 'TURBINE DECAY RATE'
        ),
        yaxis = dict(
            title = 'TURBINE INJECTION CONTROL'
        )
    )
    labels = [trace10]
    py.iplot(labels, filename=trace10)
    
    trace11 = [go.Bar(
            x=df['18 - GT Turbine decay state coefficient.'],
            y=df['16 - Fuel flow (mf) [kg/s]']
    )]
    layout = go.Layout(
        xaxis =dict(
            title = 'TURBINE DECAY RATE'
        ),
        yaxis = dict(
            title = 'FUEL FLOW'
        )
    )
    labels = [trace11]
    py.iplot(labels, filename=trace11)
    
    trace12 =[ go.Bar(
            x=df['17 - GT Compressor decay state coefficient.'],
            y=df['3 - Gas Turbine shaft torque (GTT) [kN m]']
             
        )]
    
    layout = go.Layout(
        xaxis =dict(
            title = 'COMPRESSOR DECAY RATE'
        ),
        yaxis = dict(
            title = 'TURBINE SHAFT TORQUE'
        )
    )
    labels = [trace12]
    py.iplot(labels, filename=trace12)
    

    trace13 =[ go.Bar(
            x=df['17 - GT Compressor decay state coefficient.'],
            y=df['4 - Gas Turbine rate of revolutions (GTn) [rpm]']
        )]
    layout = go.Layout(
        xaxis =dict(
            title = 'COMPRESSOR DECAY RATE'
        ),
        yaxis = dict(
            title = 'TURBINE RATE OF REVOLUTIONS'
        )
    )
    labels = [trace13]
    py.iplot(labels, filename=trace13)
    
    trace14 = [go.Bar(
            x=df['17 - GT Compressor decay state coefficient.'],
            y=df['5 - Gas Generator rate of revolutions (GGn) [rpm]']
    )]
    layout = go.Layout(
        xaxis =dict(
            title = 'COMPRESSOR DECAY RATE'
        ),
        yaxis = dict(
            title = 'GENERATOR RATE OF REVOLUTIONS'
        )
    )
    labels = [trace14]
    py.iplot(labels, filename=trace14)
    
    trace15 = [go.Bar(
            x=df['17 - GT Compressor decay state coefficient.'],
            y=df['6 - Starboard Propeller Torque (Ts) [kN]']
    )]
    layout = go.Layout(
        xaxis =dict(
            title = 'COMPRESSOR DECAY RATE'
        ),
        yaxis = dict(
            title = 'STARBOARD PROPELLER TORQUE'
        )
    )
    labels = [trace15]
    py.iplot(labels, filename=trace15)
    
    trace16 = [go.Bar(
             x=df['17 - GT Compressor decay state coefficient.'],
             y=df['7 - Port Propeller Torque (Tp) [kN]']
    )]
    layout = go.Layout(
        xaxis =dict(
            title = 'COMPRESSOR DECAY RATE'
        ),
        yaxis = dict(
            title = 'PORT PROPELLER TORQUE'
        )
    )
    labels = [trace16]
    py.iplot(labels, filename=trace16)
    
    trace17 = [go.Bar(
            x=df['17 - GT Compressor decay state coefficient.'],
            y=df['8 - HP Turbine exit temperature (T48) [C]']
    )]
    layout = go.Layout(
        xaxis =dict(
            title = 'TURBINE DECAY RATE'
        ),
        yaxis = dict(
            title = 'TURBINE EXIT TEMPERATURE'
        )
    )
    labels = [trace17]
    py.iplot(labels, filename=trace17)
    
    trace18 = [go.Bar(
            x=df['17 - GT Compressor decay state coefficient.'],
            y=df['9 - GT Compressor inlet air temperature (T1) [C]']
    )]
    layout = go.Layout(
        xaxis =dict(
            title = 'COMPRESSOR DECAY RATE'
        ),
        yaxis = dict(
            title = 'COMPRESSOR INLET AIR TEMPERATURE'
        )
    )
    labels = [trace18]
    py.iplot(labels, filename=trace18)
    
    trace19 = [go.Bar(
            x=df['17 - GT Compressor decay state coefficient.'],
            y=df['12 - GT Compressor inlet air pressure (P1) [bar]']
    )]
    layout = go.Layout(
        xaxis =dict(
            title = 'COMPRESSOR DECAY RATE'
        ),
        yaxis = dict(
            title = 'COMPRESSOR INLET AIR PRESSURE'
        )
    )
    labels = [trace19]
    py.iplot(labels, filename=trace19)
    
    trace20 = [go.Bar(
            x=df['17 - GT Compressor decay state coefficient.'],
            y=df['13 - GT Compressor outlet air pressure (P2) [bar]']
    )]
    layout = go.Layout(
        xaxis =dict(
            title = 'COMPRESSOR DECAY RATE'
        ),
        yaxis = dict(
            title = 'COMPRESSOR OUTLET AIR PRESSURE'
        )
    )
    labels = [trace20]
    py.iplot(labels, filename=trace20)
    
    trace21 = [go.Bar(
            x=df['17 - GT Compressor decay state coefficient.'],
            y=df['14 - Gas Turbine exhaust gas pressure (Pexh) [bar]']
    )]
    layout = go.Layout(
        xaxis =dict(
            title = 'COMPRESSOR DECAY RATE'
        ),
        yaxis = dict(
            title = 'TURBINE EXHAUST GAS PRESSURE'
        )
    )
    labels = [trace21]
    py.iplot(labels, filename=trace21)
    
    trace22 = [go.Bar(
            x=df['17 - GT Compressor decay state coefficient.'],
            y=df['15 - Turbine Injecton Control (TIC) [%]']
    )]
    layout = go.Layout(
        xaxis =dict(
            title = 'COMPRESSOR DECAY RATE'
        ),
        yaxis = dict(
            title = 'TURBINE INJECTION CONTROL'
        )
    )
    labels = [trace22]
    py.iplot(labels, filename=trace22)
    
    trace23 = [go.Bar(
            x=df['17 - GT Compressor decay state coefficient.'],
            y=df['16 - Fuel flow (mf) [kg/s]']
    )]
    layout = go.Layout(
        xaxis =dict(
            title = 'COMPRESSOR DECAY RATE'
        ),
        yaxis = dict(
            title = 'FUEL FLOW'
        )
    )
    labels = [trace23]
    py.iplot(labels, filename=trace23)
'''

                             ### FEATURE ENGINEERING ###
           
def featureengineering(fe):
    df = pd.read_csv(fe)
    df = df.drop('Unnamed: 0',axis=1)
    df= df.drop('Unnamed: 0.1',axis = 1)
    df= df.drop('Unnamed: 0.1.1',axis = 1)
	#df = df.drop('Unnamed: 0.1.1.1',axis=1)
    #print(df_1)
    scaler = MinMaxScaler()
    scale_data = scaler.fit_transform(df)
    scaled_data = pd.DataFrame(scale_data , columns = ['0-unnaed',
											  '1 - Lever position (lp) [ ]',
											  '2 - Ship speed (v) [knots]', 
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
                                              '15 - Turbine Injecton Control (TIC) [%]', 
											  '16 - Fuel flow (mf) [kg/s]', 
                                              '17 - GT Compressor decay state coefficient.', 
                                              '18 - GT Turbine decay state coefficient.'])
    scaled_data = scaled_data.to_csv('final_fe.csv')
    return scaled_data

                                        ### FEATURE SELECTION ###

def featureselection(selection):
        df = pd.read_csv(selection)
        #df = df.drop('INDEX1',axis= 1)
        #df= df.drop('INDEX2',axis = 1)
        df_split = np.split(df,[17],axis = 1) # splitting the dataframe to produce 2 dataframes. One with variables and
        df_variables = df_split[0]
        #df_variables1 = np.array(df_variables)
        #df_variables2 = df_variables1.reshape(1,10608)
        #df_variables = pd.DataFrame(df_variables)
        #print(df_variables)
        # other with taret variables.
        df_target = df_split[1]
        df_target1 = np.split(df_target,[1],axis = 1)
        df_target2 = df_target1[0].values
        df_target3 = df_target1[1].values
        #df_target_final  = np.concatenate((df_target2,df_target3),axis = 0)
        #df_target_final = df_target_final.reshape(len(df_target_final),1)
        #df_target_final = pd.DataFrame(df_target_final,columns=['TARGET'])
        #print(df_target)
        #print(df_split[0])
        X = df_variables
        Y = df_target2
        Y1 = df_target3
        features = df_variables.columns
        
        
        ### DIFFERENT TYPES OF FEATURE SELECTION METHODS ### 
        
                   # RECURSIVE FEATURE SELECTION - RFC
        rf = RandomForestRegressor()
        rfe = RFE(rf, n_features_to_select=3, verbose =3 )
        rfe.fit(X,Y)
        RFE_RFR= (list(map(float, rfe.ranking_)), features)
        #rfe_features = pd.DataFrame(RFE_RFR,columns =['Ranking' ,'Features'])
        print("                                                 ")
        print('Features that affect Compressor Decay State are :')
        print("                                                 ")
        print(RFE_RFR)
        
                    # EXTRA TREE REGRESSOR - REGRESSOR
        #trees = ExtraTreesRegressor(n_estimators=250,
        #                            random_state=0)

        #trees = trees.fit(X, Y)
        #ranks['ETR'] = ranking(list(map(float, trees.feature_importances_)), features, order=-1)
        #print(ranks['ETR'])
        
                                  # LASSO 
        #lasso = Lasso(alpha=.05)
        #lasso.fit(X, Y)
        #ranks['Lasso'] = ranking(np.abs(lasso.coef_), features)
        #print("/n                                                    /n")
        #print(ranks['Lasso'])
        
                                # RECURSIVE FEATURE SELECTION - RFC
        rf = RandomForestRegressor()
        rfe = RFE(rf, n_features_to_select=3, verbose =3 )
        rfe.fit(X,Y)
        RFE_RFR_2= (list(map(float, rfe.ranking_)), features)
        #rfe_features = pd.DataFrame(RFE_RFR,columns =['Ranking' ,'Features'])
        print("                                                 ")
        print('Features that affect Turbine Decay State are :')
        print("                                                 ")
        print(RFE_RFR_2)
        print("                                                 ")
        print('FINAL FEATURES AFTER FEATURE SELECTION : 3 - Gas Turbine shaft torque (GTT) [kN m]') 
        print( '4 - Gas Turbine rate of revolutions (GTn) [rpm] ,5 - Gas Generator rate of revolutions (GGn) [rpm]')
        print( '6 - Starboard Propeller Torque (Ts) [kN],7 - Port Propeller Torque (Tp) [kN]')
        print( '8 - HP Turbine exit temperature (T48) [C],9 - GT Compressor inlet air temperature (T1) [C]')
        print( '10 - GT Compressor outlet air temperature (T2) [C], 11 - HP Turbine exit pressure (P48) [bar]','13 - GT Compressor outlet air pressure (P2) [bar]')
        print( '14 - Gas Turbine exhaust gas pressure (Pexh) [bar],15 - Turbine Injecton Control (TIC) [%]')
        print( '16 - Fuel flow (mf) [kg/s]')

                                        ### MODELING ###
        

def models(model):
    df = pd.read_csv(model)
    #df =df.drop('INDEX1',axis = 1)
    #df = df.drop('INDEX2',axis = 1)
    df_split = np.split(df ,[17],axis = 1) # splitting the dataframe to produce 2 dataframes. One with variables and
    #df_split = np.split(df_split[1],[16],axis = 1 )
    df_variables = df_split[0]
    #df_variables1 = np.array(df_variables)
    #df_variables2 = df_variables1.reshape(1,10608)
    #df_variables = pd.DataFrame(df_variables)
    #print(df_variables)
    # other with taret variables.
    df_target = df_split[1]
    df_target1 = np.split(df_target,[1],axis = 1)
    df_target2 = df_target1[0].values
    df_target3 = df_target1[1].values
    X = df_variables
    Y = df_target
    
    
    train_X,test_X ,train_Y,test_Y = train_test_split(X, Y, test_size=0.30,random_state=1)
    
    mlr = LinearRegression()
    #mode1 = MultiOutputRegressor(mlr(fit(train_X , train_Y)))
    mlr.fit(train_X , train_Y)
    pickle.dump(mlr ,open('Linear_Regression.pkl', "wb" ))
    predicted_train = mlr.predict(train_X)
    predicted_test = mlr.predict(test_X)

    ### ERROR METRICS CALCULATIONS ###
    ### 1. RMSE 
    train_rmse = np.sqrt(mean_squared_error(train_Y,predicted_train))
    test_rmse =  np.sqrt(mean_squared_error(test_Y,predicted_test))      

    ### 2.MAPE 
    train_Y,predicted_train=np.array(train_Y),np.array(predicted_train) 
    test_Y,predicted_test=np.array(test_Y),np.array(predicted_test) 

    train_mape = np.mean(np.abs((train_Y - predicted_train)/train_Y))*100 
    test_mape = np.mean(np.abs((test_Y - predicted_test)/test_Y))*100 

    ### 3. MAE 
    train_mae = mean_absolute_error(train_Y , predicted_train)
    test_mae = mean_absolute_error(test_Y , predicted_test)

    ### 4.R2
    #train_Y = train_Y.reshape(1,-1)
    #predicted_train = predicted_train.reshape(1,len(predicted_train))
    #test_Y = test_Y.reshape(1,-1)
    #predicted_test = predicted_test.reshape(1,len(predicted_test))

    #train_r2 = mlr.score(train_Y , predicted_train)
    test_r2 = r2_score(test_Y , predicted_test)

    ### 5.ACCURACY 
    #train_accuracy = accuracy_score(train_Y , predicted_train)
    #test_accuracy = accuracy_Score(test_Y , predicted_test)

    #mlr_dict = {'train_rmse' :train_rmse,'test_rmse':test_rmse,'train_mape':train_mape,'test_mape':test_mape,
    #            'train_mae':train_mae ,'test_mae':test_mae,#'train_r2':train_r2,
    #            'test_r2':test_r2}
                #'train_accuracy':train_accuracy,'test_accuracy':test_accuracy}

    #df_mlr = pd.DataFrame.from_dict(mlr_dict, orient="index").to_csv('error_metrics_mlr.csv')
   
        
                               ### RANDOM FOREST REGRESSOR ###
    rfr = RandomForestRegressor(n_estimators =100, random_state = 1,n_jobs=-1)
    rfr.fit(train_X, train_Y)
    pickle.dump(rfr ,open('Random_Forest_Regression.pkl', "wb" ))
    predicted_train = rfr.predict(train_X)
    predicted_test = rfr.predict(test_X)

    ### ERROR METRICS CALCULATIONS ###
    ### 1. RMSE 
    train_rmse_rfr = np.sqrt(mean_squared_error(train_Y,predicted_train))
    test_rmse_rfr =  np.sqrt(mean_squared_error(test_Y,predicted_test))      

    ### 2.MAPE 
    train_Y,predicted_train=np.array(train_Y),np.array(predicted_train) 
    test_Y,predicted_test=np.array(test_Y),np.array(predicted_test) 

    train_mape_rfr = np.mean(np.abs((train_Y - predicted_train)/train_Y))*100 
    test_mape_rfr = np.mean(np.abs((test_Y - predicted_test)/test_Y))*100 

    ### 3. MAE 
    train_mae_rfr = mean_absolute_error(train_Y , predicted_train)
    test_mae_rfr = mean_absolute_error(test_Y , predicted_test)

    ### 4.R2
    #train_r2_rfr = rfr.score(train_Y , predicted_train)
    test_r2_rfr = r2_score(test_Y , predicted_test)


                            ### K NEAREST NEIGHBOURS ###
    knn_reg = KNeighborsRegressor(n_neighbors=11, p=2, weights="distance")
    knn_reg.fit(train_X, train_Y)
    pickle.dump(knn_reg ,open('KNN.pkl', "wb" ))
    predicted_train = knn_reg.predict(train_X)
    predicted_test = knn_reg.predict(test_X)

    ### ERROR METRICS CALCULATIONS ###
    ### 1. RMSE 
    train_rmse_knn_reg = np.sqrt(mean_squared_error(train_Y,predicted_train))
    test_rmse_knn_reg =  np.sqrt(mean_squared_error(test_Y,predicted_test))      

    ### 2.MAPE 
    train_Y,predicted_train=np.array(train_Y),np.array(predicted_train) 
    test_Y,predicted_test=np.array(test_Y),np.array(predicted_test) 

    train_mape_knn_reg = np.mean(np.abs((train_Y - predicted_train)/train_Y))*100 
    test_mape_knn_reg = np.mean(np.abs((test_Y - predicted_test)/test_Y))*100 

    ### 3. MAE 
    train_mae_knn_reg = mean_absolute_error(train_Y , predicted_train)
    test_mae_knn_reg = mean_absolute_error(test_Y , predicted_test)

    ### 4.R2
    #train_r2_knn_clf = mlp.score(train_Y , predicted_train)
    test_r2_knn_clf = r2_score(test_Y , predicted_test)

    ### 5.ACCURACY 
    #train_accuracy_knn_clf = accuracy_score(train_Y , predicted_train)
    #test_accuracy_knn_clf= accuracy_Score(test_Y , predicted_test)

    my_dict = {'train_rmse' :train_rmse,'test_rmse':test_rmse,'train_mape':train_mape,'test_mape':test_mape,
            'train_mae':train_mae ,'test_mae':test_mae,
            'test_r2':test_r2,
            'train_rmse_rfr' :train_rmse_rfr,'test_rmse_rfr':test_rmse_rfr,'train_mape_rfr':train_mape_rfr,
            'test_mape_rfr':test_mape_rfr,
            'train_mae_rfr':train_mae_rfr ,'test_mae_rfr':test_mae_rfr,
            'test_r2_rfr':test_r2_rfr,
            'train_rmse_knn_reg' :train_rmse_knn_reg,'test_rmse_knn_reg':test_rmse_knn_reg,
            'train_mape_knn_reg':train_mape_knn_reg,'test_mape_knn_reg':test_mape_knn_reg,
            'train_mae_knn_reg':train_mae_knn_reg ,'test_mae_knn_reg':test_mae_knn_reg,
            'test_r2_knn_clf':test_r2_knn_clf}
    
    df_error_metrics = pd.DataFrame.from_dict(my_dict, orient="index").to_csv('error_metrics.csv')
    
    csv = pd.read_csv('error_metrics.csv')
    #csv = np.array(csv[1])
    csv = np.split(csv , [1],axis=1)
    csv[1]
    csv = np.array(csv[1])

    model_names = ['Linear_Regression','Random_Forest_Tree','K_Nearest_Neighbor']
    Linear_Regression = csv[0] - csv[1]
    Random_Forest_Tree = csv[7] - csv[8]
    K_Nearest_Neighbor = csv[14] - csv[15]
    rmse=[Linear_Regression,Random_Forest_Tree,K_Nearest_Neighbor]

    Linear_Regression = csv[4] - csv[5]
    Random_Forest_Tree = csv[9] - csv[10]
    K_Nearest_Neighbor = csv[18] - csv[19]
    mae=[Linear_Regression,Random_Forest_Tree,K_Nearest_Neighbor]

    Linear_Regression = csv[6]
    Random_Forest_Tree = csv[11] 
    K_Nearest_Neighbor = csv[20]
    r2=[Linear_Regression,Random_Forest_Tree,K_Nearest_Neighbor]

    #print('The model which has high RMSE value and the worst model is' , min(model_names) ,'value' ,min(mae))

    rank_1 =  'least RMSE value and the best model is',max(model_names) 
    rank_1_1 = max(rmse)
    rank_2 =  'high RMSE value and the worst model is',min(model_names) 
    rank_2_1 = min(rmse)
    rank_3 =  'least MAE value and the best model is',max(model_names)  
    rank_3_1 = max(mae)
    rank_4 =  'high RMSE value and the worst model is',min(model_names)
    rank_4_1 = min(mae)
    rank_5 =  'high R2 score value and the best model is',max(model_names) 
    rank_5_1 = max(r2)
    rank_6 =  'low R2 score value and the worst model is',min(model_names)
    rank_6_1 = min(r2)

    ranks =[]
    i = 1

    ranks.append(rank_1)
    ranks.append(rank_2)
    ranks.append(rank_3)
    ranks.append(rank_4)
    ranks.append(rank_5)
    ranks.append(rank_6)
    ranks.append(rank_1_1)
    ranks.append(rank_2_1)
    ranks.append(rank_3_1)
    ranks.append(rank_4_1)
    ranks.append(rank_5_1)
    ranks.append(rank_6_1)
    ranks = np.array(ranks)
    ranks = ranks.reshape(12,1)
    #ranks = pd.DataFrame(ranks)
    ranks = np.split(ranks , [6])
    ranks_csv_0 = pd.DataFrame(ranks[0] , columns=['MODELS'])
    ranks_csv_1 = pd.DataFrame(ranks[1] , columns=['VALUES'])
    ranks_csv = ranks_csv_0.join(ranks_csv_1)
    ranks_csv.to_csv('compiled_ranks.csv' , index = False)
    
    filenames = ['error_metrics.csv','compiled_ranks.csv']
    combined_final = ([ pd.read_csv(f) for f in filenames ])
    combined_final = combined_final[0].join(combined_final[1])
    combined_final.to_csv('compiled_final.csv',index=False)


make_pipeline(cleaning('clean.csv'),#eda('clean_data1.csv'),
              featureengineering('clean_data1.csv'),
              featureselection('final_fe.csv'),
              models('final_fe.csv'))#,s3upload()) 

                                             
                                            ### S3 UPLOAD ###

buck_name="dhanisha" #enter bucket name
Input_location = 'us-east-1'
S3_client = boto3.client('s3', Input_location, aws_access_key_id= Access_key, aws_secret_access_key= Secret_key)

if Input_location == 'us-east-1':
        S3_client.create_bucket(
            Bucket=buck_name
        )
else:
    S3_client.create_bucket(
            Bucket=buck_name,
            CreateBucketConfiguration={'LocationConstraint': Input_location},
        )

S3_client.upload_file('compiled_final.csv', buck_name, 'compiled_final.csv')
S3_client.upload_file('Linear_Regression.pkl', buck_name, 'Linear_Regression.pkl')
S3_client.upload_file('Random_Forest_Regression.pkl', buck_name, 'Random_Forest_Regression.pkl')
S3_client.upload_file('KNN.pkl', buck_name, 'KNN.pkl')

