
# coding: utf-8

# In[31]:


#calling libraries

import  urllib
import bs4
from bs4 import BeautifulSoup as bs
import os
import pandas
from pandas import DataFrame as df
import argparse
import csv
import zipfile 
import sys
from zipfile import *
import logging
import boto3

#create logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG) #setting level to debug

#creates file handler which logs all the level messages
fh = logging.FileHandler('Log_file1.log' , mode = 'w')
fh.setLevel(logging.DEBUG) #setting level to handler to debug

formatter = logging.Formatter('[%(asctime)s] %(levelname)8s --- %(message)s' + 
                             '(%(filename)s:%(lineno)s)' ,datefmt = '%Y-%m-%d %H:%M:%S')

if (logger.hasHandlers()):
    logger.handlers.clear()
    
#if handler_console is none :
handler_console = logging.StreamHandler(stream = sys.stdout)

#then add it back 
handler_console.setFormatter(formatter)
handler_console.setLevel(logging.INFO)
logger.addHandler(handler_console)

fh.setFormatter(formatter)
logger.addHandler(fh)

#Taking inputs from commandline and validating if its empty





 #importing the parser library
parser = argparse.ArgumentParser(description='Please enter the values of below :') #creating a parser
parser.add_argument("--CIK", help = "Enter the CIK of a particular company")
parser.add_argument("--Accession_number", help = "Enter Accession_number of the same company")
parser.add_argument("--Access_key", help = "Enter Access_key of your amazon acc")
parser.add_argument("--Secret_key", help = "Enter Secret_key of your amazon acc")
parser.add_argument("--Input_location", help = "Enter location")
#parser.add_argument("--Location", help = "Enter location to create bucket")
args = parser.parse_args()#parser.add_argument("--s3loc",help="put the region you want to select for amazon s3")
# parse_args() will typically be called with no arguments, and the ArgumentParser will automatically determine the command-line arguments from sys.argv.
if not args.CIK:
    logger.warning("CIK has been not provided,exiting the system")
    sys.exit(0)
if not args.Accession_number:
    logger.warning("Accession_number has been not provided,exiting the system")
    sys.exit(0)      
if  not args.Access_key:
    logger.warning("AccessKey input has been not provided,exiting the system")
    sys.exit(0)
if not args.Secret_key:
    logger.warning("SecretKey input has been not provided,exiting the system")
    sys.exit(0)
if not args.Input_location:
    logger.warning("Location has been not provided,exiting the system")
    sys.exit(0)
 

CIK=args.CIK
Accession_number=args.Accession_number
Access_key=args.Access_key
Secret_key=args.Secret_key
Input_location=args.Input_location
#if not args.Location:
 #   logger.warning("Location input has been not provided,exiting the system")
  #  sys.exit(0)
#printing inputs to the console and into the log file
logger.info("CIK=" + args.CIK)
logger.info("Accession_number=" + args.Accession_number)
logger.info("Access_key="+ args.Access_key)
logger.info("Secret_key="+ args.Secret_key)
logger.info("Input_location="+ args.Input_location)

#automating url

url = ("https://www.sec.gov/Archives/edgar/data/")

new_cik = CIK.lstrip('0')
new_accession_number = Accession_number.replace("-","")
new_url = (url + new_cik + "/" + new_accession_number + "/" + Accession_number + "-" + "index.html")
logger.info(new_url)

#retrieving web page using above url
try :
    urllib.request.urlopen(new_url)  
    logger.info("Url request is successful")
except urllib.request.HTTPError as e:
    # throw error when cik and Accession no is invalid and exit the code
    logger.error("cik and Accession number is invalid") 
    logger.exception("HTTP url not found" + str(e))
    exit(0)

#reading and parsing the web page using beautifulsoup
new_url_1 = urllib.request.urlopen(new_url).read()
new_url_2 = bs(new_url_1 , 'html.parser')


#finding the form-10q
#finding the form-10q
tbl = new_url_2.findAll('table',{'summary':'Document Format Files'})
tbl1 = tbl[0] # cob=nverting list into table
tr1= tbl[0].findAll('tr') #extracting all tr of the the table

i=0
url=""
for i in range(1,len(tr1)):
    td_list= tr1[i].findAll('td')
    for j in range(0, len(td_list)+1):
        if (j==3):
            if(td_list[j].text == "10-Q"):
                url= td_list[j-1].text
                break
        
                

#retrieve all the possible 10-Q form and extract its tables

final_url = "https://www.sec.gov/Archives/edgar/data/" + new_cik + "/" + new_accession_number +"/"+ url 
scraping_page = urllib.request.urlopen(final_url).read()
logger.info("10-q form request is successful")
scrape = bs(scraping_page,"html.parser")

# extracting the tables from 10-q form  
logger.info("Table extraction started")
if not os.path.exists('FINAL_TABLES'):
    try:
        os.makedirs('FINAL_TABLES') 
    except OSError as exc: # Guard against race condition
            logger.error("OSError" + str(exc))
#extracting all div yag
scrape_page = scrape.select('div table')
rawlist_tables=[]
for tables_1 in scrape_page:
    for rows in tables_1.find_all('tr'):
        number_tables=0
        for col in rows.findAll('td'):
            if('$' in col.get_text() or '%' in col.get_text()):
                rawlist_tables.append(tables_1)
                number_tables=1;
                break;
                if(number_tables==1):
                    break;
                                


for tables_2 in rawlist_tables:
    final_list = []
    for rows_1 in tables_2.find_all('tr'):
        row=[]
        for columns_1 in rows_1.findAll('td'):
            para = columns_1.find_all('p')               
            if len(para)>0:
                for para_1 in para:
                    new_para = para_1.get_text().replace("\n"," ") 
                    new_para_1 = new_para.replace("\xa0","")                 
                    row.append(new_para_1)
            else :
                new_col=columns_1.get_text().replace("\n"," ")
                new_col_2=new_col.replace("\xa0","")
                row.append(new_col_2)

        final_list.append(row)

    with open(os.path.join('FINAL_TABLES' ,  'DATA' + str(rawlist_tables.index(tables_2)) + '.csv'), 'w') as csv_file:
         csv_writer = csv.writer(csv_file)
         csv_writer.writerows(final_list)
         logger.info("Table has been extracted in csv format")


# zipping all the csv files
zip_file = zipfile.ZipFile('Extracted_tables.zip' , 'w',zipfile.ZIP_DEFLATED)
for tables_2 in rawlist_tables:
    zip_file.write(os.path.join('FINAL_TABLES' ,  'DATA' + str(rawlist_tables.index(tables_2)) + '.csv'))
    logger.info("All files are zipped")

zip_file.close()

 
#uploading files to amazon s3 
'''BucketAlreadyOwnedByYou errors will only be returned outside of the US Standard region. 
Inside the US Standard region (i.e. when you don't specify a location constraint), attempting to recreate a bucket you already own will succeed.'''
logger.info("Uploading files to amazon")
try:

    buck_name="ads-part1-assignment"

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
    logging.info("connection successful")
    S3_client.upload_file("Extracted_tables.zip", buck_name, "Extracted_tables.zip")
    S3_client.upload_file("Log_file1.log", buck_name, "Log_file1.log")
    logging.info("Files uploaded successfully")
except Exception as e:
    logging.error("Error uploading files to Amazon s3" + str(e))

# In[ ]:



     

