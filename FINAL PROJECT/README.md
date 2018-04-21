# PREDICTING AIRLINE DELAYS 
 ## WEEK 1 :
 ### STEP 1:
 #### <b>EXPLORATORY DATA ANALYSIS :</b>
 <i>
 1.The dataset is uploaded to the <b>SQL server</b>.<br>
 2.It is then accessed by <b>Tableau</b> which is used to analyse the dataset.<br>
 3.The link to the viz is given below :<br></i>
 <i><b>https://public.tableau.com/shared/WJH6JSS4B?:display_count=yes</b><br></i>
 
 ### STEP 2:
 #### <b>USER INTERFACE USING SHINY :</b>
 <i>
 
 1.The application created will ask the user to input the 'Date' (of travel) , 'Starting'(origin airport) , 'Destination'(destination        airport).<br>
 2.The UI can be published by connecting the Rstudio to <b>Shinyapps.io</b> or <b>Rstudio connect</b>.<br>
 3.Here we are using shinyapps.io to publish the application.<br>
 4.The link to our application is :<b>https://ads-final-project.shinyapps.io/USER-INTERFACE/</b> ( modifications are yet to be made).<br>
 5.Once the user inputs all the necessary data , the user will get the output of the percentage of delay that has happened in the past on    that day for that particular origin and destination.<br></i>

## WEEK 2 :
### STEP 1:
#### <b>PREDICTION OF AIRLINE DELAYS :</b>
<i>
 1.The dataset in the form of csv is imported from the AWS cloud . The dataset as of now contains only 2008's data. (We are trying to include more years to that).<br>
 2.Three outputs are given to the user.<br>
 a) Is the flight delayed or not? - This is predicted using <b>Naive Bayes classification</b> algorithm with a accuracy score of 96%.<br>
   b) How much is the flight delayed at the origin airport and what is the arrival delay? - Predicted using <b>RandomForestRegressor </b>withe r2 score of 99%.<br
   c) What type of delay is it ? - There are four delays [Carrier delay , Security delay , Weather delay , NAS delay] . This is predicted using <b>RandomForestClassifier</b> with accuracy of 89%.<br>
 </i>
 
 ## WEEK 3 :
 1. Tuning of the parameters.
 2. Add more datasets to provide efficient results.
