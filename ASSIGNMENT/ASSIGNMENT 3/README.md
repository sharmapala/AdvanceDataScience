# ASSIGNMENT 3 
### PART 1 : PIPELINING 
1. The given dataset was accessed from the aws platform.
2. Data cleaning , feature engineering and feature selection was performed on the dataset.
3. 3 types of models were deployed(<b>Linear Regression , Random Forest Regressor,K-Nearest Neighbors</b>) to make prediction.
4. Error metrics such as <b>RMSE, R2, MAPE and MAE</b> were calculated and stored in a csv file along with the ranking of models based on the error metrics value.
5. All the models were make into <b>pickle files</b>.
6. The pickle files and the error metrics csv file was then pushed to aws s3 bucket.
7. The whole process was executed using <b>sklearn pipeline</b>.
8. The jupyter file was then dockerised and the image was pushed to the <b>docker hub</b> which can be pulled and executed at any time.
