# PREDICTING AIRLINE DELAYS 
 ## WEEK 1 :
 ### DIFFICULTIES :
 <i>
 1. Initially in the projetc proposal , we had proposed that spark will be included but later when we started working on the project , setting up of the cluster was very difficult.<br>
 2. Tried cloudera and Hortonworks sandbox but due to technical difficulties it didn't boot up in our system.<br>
 3.However we uploaded the file to hdfs from cloudera terminal , but pulling it from the HDFS was difficult.<br>
 Thus we changed the method of implementation of the project. </i><br>
 
 ### STEP 1:<br>
 #### <b>EXPLORATORY DATA ANALYSIS :</b>
 <i>
 1.The dataset is uploaded to the <b>SQL server</b>.<br>
 2.It is then accessed by <b>Tableau</b> which is used to analyse the dataset.<br>
 3.The link to the viz is given below :<br></i>
 <i><b>https://public.tableau.com/shared/WJH6JSS4B?:display_count=yes</b><br></i>

## WEEK 2 :
### STEP 1:
#### <b>PREDICTION OF AIRLINE DELAYS :</b>
<i>
 1.The dataset in the form of csv is imported from the AWS cloud . The dataset as of now contains only 2008's data. (We are trying to include more years to that).<br>
 2.Three outputs are given to the user.<br>
 a) Is the flight delayed or not? - This is predicted using <b>Naive Bayes classification</b> algorithm with a accuracy score of 96%.<br>
   b) How much is the flight delayed at the origin airport and what is the arrival delay? - Predicted using <b>RandomForestRegressor </b><br>
   c) What type of delay is it ? - There are four delays [Carrier delay , Security delay , Weather delay , NAS delay] . This is predicted using <b>RandomForestClassifier</b> and also the value of each is predicted using Random Forest Regresor.<br>
 d)  Working on the app building on agile basis.
 </i>
 
 ## WEEK 3 :
 <i>
 <b>
 1. We shifted from AWS Cloud to Google Cloud for storing our dataset. And now we are fetching our dataset from GCP.<br>
 2. Added some more years of data in the dataset to get more insight and better classification results.</b><br>
 3.The prediction model pipeline which was running on local previously is now deployed on AWS cloud up and running on EC2 instance - through ssh terminal. </b><br> (The instance might change, just a trial basis)
 4.Pickel files of all the three classification models used for all three classifications:</b><br>
 a. Delayed or not (https://s3-us-west-2.amazonaws.com/team3/delay.pkl)</b><br>
 b. Category of delay (https://s3-us-west-2.amazonaws.com/team3/delay_type_value.pkl)</b><br>
 c. Average departure or Arrival Delay (https://s3-us-west-2.amazonaws.com/team3/delay_value.pkl)
 are pushed to s3 bucket.</b><br>
 3.The python script is dockerized and the image will be pulled from a EC2 instance.</b><br>
 4. On Front-End, A fully developed Web App for the user to use with authentication via Login using Flask is deployed on the EC instance up and running on  ec2-54-191-117-224.us-west-2.compute.amazonaws.com </b><br>
 5. Trying to implement Dask and bokeh.</b><b><br>
 6. RESTAPIs are used to make the web App more user friendly.</b><B><br>
 7. Bokeh implemnted successfully. Dask only on local single machine. </b><br>
 
 
</i>
