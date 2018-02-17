PROBLEM 1:
In this part of the question, we have automated the generation of URL based on the CIK and Accession number.
Once the URL is formed, all the 10-q links will be parsed and tables from the forms will be scraped and stored as individual tables which we have later zipped and pushed it to S3 bucket. 
The whole process is automated using Docker as the last step of this problem.

PROBLEM 2:
This part deals with the extraction of the log files present on the link given below:
https://www.sec.gov/dera/data/edgar-log-file-data-set.html
and then after downloading the zipped files inside a directory, these files are unzipped to access the CSVs and the data inside them.
A series of pre-processing and data wrangling is applied on the 1st day(next one if the previous is not available) of each month of a particular year. 
The data is converted to a data frame and missing data is handled first , then the summary metrics in both statistical and graphical form is calculated. The analysis ends with calculating the anomalies in the data, if any.
