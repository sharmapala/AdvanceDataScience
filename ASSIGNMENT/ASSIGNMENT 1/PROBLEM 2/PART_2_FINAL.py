
# coding: utf-8

# In[1]:


import urllib.request
import zipfile
import os
import pandas as pd
import logging # for logging
import shutil #to delete the directory contents
import glob
import boto3
import sys
from itertools import groupby
import argparse
import numpy as np
from statistics import mode
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
    # create logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG) # setting level to debug 

    # create file handler which logs all the level messages
fh = logging.FileHandler('Log_file2.log', mode='w') 
fh.setLevel(logging.DEBUG) # set level of handler to debug 

formatter = logging.Formatter('[%(asctime)s] %(levelname)8s --- %(message)s ' +
                                  '(%(filename)s:%(lineno)s)',datefmt='%Y-%m-%d %H:%M:%S')
    
if (logger.hasHandlers()):
    logger.handlers.clear()

#if handler_console is None:
handler_console = logging.StreamHandler(stream=sys.stdout)
            
# then add it back
handler_console.setFormatter(formatter)
handler_console.setLevel(logging.INFO)
logger.addHandler(handler_console)
    

fh.setFormatter(formatter)
logger.addHandler(fh)

#--------------------------------------------------------------------------------------------------#

 #importing the parser library
parser = argparse.ArgumentParser(description='Please enter the values of below :') #creating a parser
parser.add_argument("--year", help = "Enter the year for which you need log file")
parser.add_argument("--accessKey", help = "Enter access key of your amazon acc")
parser.add_argument("--secretKey", help = "Enter secret key of your amazon acc")
#parser.add_argument("--Location", help = "Enter location to create bucket")
args = parser.parse_args()#parser.add_argument("--s3loc",help="put the region you want to select for amazon s3")
# parse_args() will typically be called with no arguments, and the ArgumentParser will automatically determine the command-line arguments from sys.argv.
if not args.year:
    logger.warning("Year input has been not provided,exiting the system")
    sys.exit(0)
if  not args.accessKey:
    logger.warning("AccessKey input has been not provided,exiting the system")
    sys.exit(0)
if not args.secretKey:
    logger.warning("SecretKey input has been not provided,exiting the system")
    sys.exit(0)
#if not args.Location:
    logger.warning("Location input has been not provided,exiting the system")
    sys.exit(0)

accessKey = args.accessKey
secretKey = args.secretKey
#Location = args.Location

year = args.year
logger.info(year)
years = range(2003,2018)
if int(year) not in years:
    logger.error("Year is not in the valid range,exiting the system")
    sys.exit(0)
try:
    if not os.path.exists('zippedfiles_zip'):
        os.makedirs('zippedfiles_zip', mode=0o777)
        logger.info('Zipped file directory created!!')
    else:
        shutil.rmtree(os.path.join(os.path.dirname("__file__"),'zippedfiles_zip'), ignore_errors=False)
        os.makedirs('zippedfiles_zip', mode=0o777)
        logger.info('Zipped file directory created!!')
    
    if not os.path.exists('unzippedfiles'):
        os.makedirs('unzippedfiles', mode=0o777)
        logger.info('UnZipped file directory created!!')
    else:
        shutil.rmtree(os.path.join(os.path.dirname("__file__"), 'unzippedfiles'), ignore_errors=False)
        os.makedirs('unzippedfiles', mode=0o777)
        logger.info('UnZipped file directory created!!')
except Exception as e:
    logger.exception(str(e))

#------------------------------------------------------------------------------------------------#
try:
    Quaters = { 'Qtr1' : ['01','02','03'], 'Qtr2' : ['04','05','06'],'Qtr3' : ['07','08','09'], 'Qtr4' : ['10','11','12']}
    days= range(1,32)
    for key,value in Quaters.items():
        for val in value:
            for d in days:
                url= 'http://www.sec.gov/dera/data/Public-EDGAR-log-file-data/' + str(year) + '/'+ str(key)+ '/log'+ str(year)+str(val)+str(format(d,'02d')) +'.zip'
                urllib.request.urlretrieve(url, r'zippedfiles_zip/'+url[-15:])
                logger.info("Retrieving zipped log file")
                if os.path.getsize('zippedfiles_zip/'+url[-15:]) <= 4515: #catching empty file
                    os.remove('zippedfiles_zip/'+url[-15:])
                    logger.info("Log file is not present for "+ str(d) + " day")
                    continue
                break
except:
    logging.warning("No data present for the current month, performing analysis on previous months")
    
#------------------------------------------------------------------------------------------------#
try:
    zip_files = os.listdir('zippedfiles_zip')
    for f in zip_files:
        z = zipfile.ZipFile(os.path.join('zippedfiles_zip', f), 'r')
        for file in z.namelist():
            if file.endswith('.csv'):
                z.extract(file, r'unzippedfiles')
                logger.info(file +' successfully extracted to folder: unzippedfiles.')
except Exception as e:
        logger.error(str(e))
#----------------------------------------------------------------------------------------------#

try:
        allFiles = glob.glob(r'unzippedfiles' + "/*.csv")
except Exception as e:
     logger.error(str(e) + "No csv files present")

#----------------------------------------------------------------------------------------------#
#HANDLING MISSING DATA

if not os.path.exists('Graphical_images'):
    os.makedirs('Graphical_images', mode=0o777)
    logger.info('Graphical_images directory created!!')
else:
    shutil.rmtree(os.path.join(os.path.dirname("__file__"),'Graphical_images'), ignore_errors=False)
    os.makedirs('Graphical_images', mode=0o777)
    logger.info('Graphical_images directory created!!')

i=1
for file_ in allFiles:
    data = pd.read_csv(file_) #reading the csv files one by one from all the extracted files
    logger.info("Calculated the missing values in each coloumn")
    print(data.isnull().sum())
    logger.info("Finding the NaN values in each coloumn")
    logger.info("Working on the Browser Coloumn")
    a = ['mie','fox','saf','chr','sea', 'opr','oth','win','mac','lin','iph','ipd','and','rim','iem']
    [len(list(group)) for key, group in groupby(a)] 
    logger.info("Grouping the values of all the browsers and storing in a dataframe")
    df = pd.DataFrame(data,columns = ['browser'])
    d = df.apply(pd.value_counts)
    logger.info("Counting the frequency of each browser type used in descending order")
    list(d.index)
    d.index[0]
    logger.info("Selecting the brower at the index 0(as it will be the maximum used browser)")
    data['browser'].replace(np.nan,d.index[0], inplace = True) #Replacing the NaN values in the browser colomun with the max used browser
    data.isnull().sum()
    logger.info("confirming that no NaN values are present on the browser coloumn")
    logger.info("Working on the Size Coloumn")
    logger.info("Replacing the file size for ext : txt, by the mean of all the file size corresponding to txt")
    s = data[['extention','size']].groupby(data['extention'].str.contains('txt'))['size'].mean().reset_index(name='mean').sort_values(['mean'],ascending=False)
    data.loc[(data['size'].isnull()) & (data['extention'].str.contains('txt'))]
    data.loc[(data['size'].isnull()) & (data['extention'].str.contains('txt'))] = s
    data.loc[(data['size'].isnull()) & (data['extention'].str.contains('txt'))]
    data.reset_index(drop = True)
    logger.info("Replacing the file size with NaN values for ext : htm, by the mean of all the file size corresponding to htm") 
    g = data[['extention','size']].groupby(data['extention'].str.contains('htm'))['size'].mean().reset_index(name='mean').sort_values(['mean'],ascending=False)
    data.loc[(data['size'].isnull()) & (data['extention'].str.contains('htm'))]
    data.loc[(data['size'].isnull()) & (data['extention'].str.contains('htm'))] = g
    data.loc[(data['size'].isnull()) & (data['extention'].str.contains('htm'))]
    data.reset_index(drop=True)
    logger.info("Replacing the file size with NaN values for ext : xml, by the mean of all the file size corresponding to xml")
    h = data[['extention','size']].groupby(data['extention'].str.contains('xml'))['size'].mean().reset_index(name='mean').sort_values(['mean'],ascending=False)
    data.loc[(data['size'].isnull()) & (data['extention'].str.contains('xml'))]
    data.loc[(data['size'].isnull()) & (data['extention'].str.contains('xml'))] = h
    data.loc[(data['size'].isnull()) & (data['extention'].str.contains('xml'))]
    data.reset_index(drop=True)
    logger.info("To check how many NaN values are remaining ")
    logger.info("Replacing the file size for rest of the files with the mean of file size of txt extension, as it is the max used")
    data.loc[data['size'].isnull()] = s
    print(data.isnull().sum())
    logger.info("Working on all other coloumns")
    logger.info("If cik,Accession,ip,date are empty fields drop the records")
    data.dropna(subset=['cik'],inplace=True)
    data.dropna(subset=['accession'],inplace=True)
    data.dropna(subset=['ip'],inplace=True)
    data.dropna(subset=['date'],inplace=True)
    data.dropna(subset=['time'],inplace=True)
    logger.info("Calculating the max categorical value in other coloumns( code, zone,extention,idx,find) and filling the NaNs")
    data['code'].fillna(data['code'].max(),inplace=True)
    data['zone'].fillna(data['zone'].max(),inplace=True)
    data['extention'].fillna(data['extention'].max(),inplace=True)
    data['idx'].fillna(data['idx'].max(),inplace=True)
    data['find'].fillna(data['find'].max(),inplace=True)
    
    logger.info("Filling empty values with Categorical Values for coloumns (norefer,noagent,nocrawler)")
    data['norefer'].fillna(1,inplace=True)
    data['noagent'].fillna(1,inplace=True)
    data['crawler'].fillna(0,inplace=True)
    print(data.isnull().sum())
    logger.info("Missing data is handled successfully")
#---------------------------------------------------------------------------------------------------#
#SUMMARY METRICS
    logger.info("Calculating Summary metrics of clean data")
    data.describe()
    data.reset_index(drop = True)
    #logger.info("To calculate which class of IP addresses(Class A, Class B , Class C) has done the maximum EDGAR fillings using mode")
    #df = pd.DataFrame(data,columns = ['ip'])
    #ClassAlist = []
    #ClassBlist = []
    #Others = []
    #for i in range(0,len(df.index)):
     #   a = df.iloc[i]
      #  octects = a.str.split('.',expand = True)
       # o = octects.iloc[0,0]
        #for o in range(0,128):
         #   ClassAlist.append(int(o))
        #for o in range(128,183):
         #   ClassBlist.append(int(o))
        #for o in range(183,256):
         #   Others.append(int(o))
    #if len(ClassAlist) > (len(ClassBlist) | len(Others)):
     #   logger.info("The Companies who have filled maximum are the ones having ClassA ips ")
    #if len(ClassBlist) > (len(ClassAlist) | len(Others)):                                                                       
     #   logger.info("The Companies who have filled maximum are the ones having ClassB ips ")
    #if len(Others) > (len(ClassBlist) | len(ClassAlist)):
     #   logger.info("The Companies who have filled maximum are the ones having ClassC ips ")                                                                            


    logger.info("Mean and Median sizes for each Browser")
    brow_df = data.groupby('browser').agg({'size':['mean', 'median'],'crawler': len})
    brow_df.columns = ['_'.join(col) for col in brow_df.columns]
    data.reset_index(drop=True)
    print(brow_df)
                                                                                  
#To find out the 15 top searched CIKs 
    cik_df = pd.DataFrame(data, columns = ['cik'])
    d = cik_df.apply(pd.value_counts)
    logger.info("Top 15 most searched CIKs with the count")                                                                            
    d.head(15)
    data.reset_index(drop=True)
                    
#Compute distinct count of ip per month i.e. per log file
    ipcount_df = df['ip'].nunique()
    logger.info("Compute distinct count of ip per month i.e. per log file")
    print(ipcount_df)
                                                                                  
#Computing the count of status code on the basis of ip
    StCo_count=data[['code','ip']].groupby(['code'])['ip'].count().reset_index(name='count')
    logger.info("Computing the count of status code on the basis of ip")
    print(StCo_count)
    data.reset_index(drop=True)
                    
#Everything on per day basis
    #1. Average of Size 
    Avg_size=data[['date','size']].groupby(['date'])['size'].mean().reset_index(name='mean')
    logger.info("Average of file size is computed")
    print(Avg_size)
    #2. Number Of Requests
    Req_day=data[['date','ip']].groupby(['date'])['ip'].count().reset_index(name='count')
    logger.info("Number of request per day is computed")
    print(Req_day)
#Mean of file size on the basis of code status
    Mean_size=data[['code','size']].groupby(['code'])['size'].mean().reset_index(name='mean')
    logger.info("Mean of file size on the basis of code status")
    print(Mean_size)

    logger.info("Summary metrics computed succesfully!!")
    #file = '/log'+ str(year)+str(val)+str(format(d,'02d'))
    #filename="log"+str(year)+str(val)+str(format(d,'02d'))
#--------------------------------------------------------------------------------------------------#
#GRAPHICAL REPRESENTATION 
#graph of no of status codes by browser
    try:
    	 
        logger.info("graphical analysis started")
        Num_of_codes=data[['browser','code']].groupby(['browser'])['code'].count().reset_index(name = 'count_code').sort_values(['count_code'],ascending=False)
        data.reset_index(drop = True)
        print(Num_of_codes)
        x= np.array(range(len(Num_of_codes)))
        y= Num_of_codes['count_code']
        xticks1 = Num_of_codes['browser']
        plt.xticks(x,xticks1)
        plt.bar(x,y)
        plt.title('Count of status code for all the browsers')
        plt.ylabel('Count of codes')
        plt.xlabel('Browsers')
        plt.savefig('Graphical_images/countsperbrowser'+ str(i) +'.png',dpi=100)
        #plt.savefig(os.path.join('Graphical_images',str(val),'countsperbrowser.png'),dpi=100)
        #plt.show()
        plt.clf()
        logger.info("graphical analysis end")
    except Exception as e:
        logger.error(str(e))
        logging.error("Error plotting the graph ")
    
#graph for max cik(10) by IP used
    try:
        logger.info("graphical analysis started")
        Num_of_CIKs=data[['cik','ip']].groupby(['cik'])['ip'].count().reset_index(name='count').sort_values(['count'],ascending=False).head(10)
        data.reset_index(drop=True)
        print(Num_of_CIKs)
        x = np.array(range(len(Num_of_CIKs)))
        y = Num_of_CIKs['count']
        xticks2 = Num_of_CIKs['cik']
        plt.xticks(x, xticks2)
        plt.bar(x,y)
        plt.title('Top 10 CIKs by IPs')
        plt.ylabel('Count of IPs')
        plt.xlabel('CIK-')
        #plt.savefig(os.path.join('Graphical_images',str(val),'CIKsbyIPcount.png'),dpi=100)
        plt.savefig('Graphical_images/CIKsbyIPcount'+ str(i) +'.png',dpi=100)
        #plt.show()
        plt.clf()
        logger.info("graphical analysis end")
    except Exception as e:
        logger.error(str(e))
        logging.error("Error plotting the graph ")

#Graph of Mean of file size on the basis of code status

    try:
        Mean_size=data[['code','size']].groupby(['code'])['size'].mean().reset_index(name='mean').sort_values(['mean'],ascending=False)
        data.reset_index(drop=True)
        print(Mean_size)
        x = np.array(range(len(Mean_size)))
        y = Mean_size['mean']
        xticks3 = Mean_size['code']
        plt.xticks(x, xticks3)
        plt.bar(x,y)
        plt.title('filesize by codes')
        plt.ylabel('mean size')
        plt.xlabel('Code')
        #plt.savefig(os.path.join('Graphical_images',str(val),'MeanSizeByCode.png'),dpi=100)
        plt.savefig('Graphical_images/MeanSizeByCode'+str(i) +'.png',dpi=100)
        #plt.show()
        plt.clf()
    except Exception as e:
        logger.error(str(e))
        logging.error("Error plotting the graph ")
#Graph for average file size by extension
    try:
        Avg_size=data[['extention','size']].groupby(['extention'])['size'].mean().reset_index(name='mean').sort_values(['mean'],ascending=False).head(20)
        data.reset_index(drop=True)
        print(Avg_size)
        x = np.array(range(len(Avg_size)))
        y = Avg_size['mean']
        xticks4 = Avg_size['extention']
        plt.xticks(x, xticks4)
        plt.bar(x,y)
        plt.title('Avg File size by extention')
        plt.ylabel('MeanFileSize')
        plt.xlabel('Extention')
        #plt.savefig(os.path.join('Graphical_images',str(val),'filesizebyextention.png'),dpi=100)
        plt.savefig('Graphical_images/filesizebyextention'+str(i) +'.png',dpi=100)
        #plt.show()
        plt.clf()
    except Exception as e:
        logger.error(str(e))
        logging.error("Error plotting the graph ")
#----------------------------------------------------------------------------------------------#
#ANAMOLIES IN FILESIZE
#STATISTICAL APPROACH(Using)
 #   def feature_normalize(data):
  #  mu = np.mean(data,axis=0)
   # sigma = np.std(data,axis=0)
    #return (data - mu)/sigma
#def estimateGaussian(data):
 #   mu = np.mean(data, axis=0)
  #  sigma = np.cov(data,rowvar=False)
   # return mu, sigma
#Graphical Approach
#Anomalies in FileSize
    try:
        logger.info("Anomalies analysis started")
        data.boxplot(column='size',vert=True,sym='',whis=10,showfliers=False)
        plt.xticks(rotation=70)
        plt.title('Anomalies displayed on the file size')
        plt.ylabel('size')
        #plt.savefig(os.path.join('Graphical_images',str(val),'Anomalies.png'),dpi=100)
        plt.savefig('Graphical_images/'+'Anomalies'+ str(i) +'.png',dpi=100)
        #plt.show()
        logger.info("Anomalies analysis ended")
    except Exception as e:
        logger.error(str(e))
        logging.error("Error plotting the graph ")

    i = i+1

  
#Making a zip file having the log file and the Graphical_images folder
def make_zipfile(output_filename, source_dir):
    relroot = os.path.abspath(os.path.join(source_dir, os.pardir))
    with zipfile.ZipFile(output_filename, "w", zipfile.ZIP_DEFLATED) as zip:
        for root, dirs, files in os.walk(source_dir):
            # add directory (needed for empty dirs)
            zip.write(root, os.path.relpath(root, relroot))
            for file in files:
                filename = os.path.join(root, file)
                if os.path.isfile(filename): # regular files only
                    arcname = os.path.join(os.path.relpath(root, relroot), file)
                    zip.write(filename, arcname)

make_zipfile(ADSAssign1Part2.zip,Graphical_images)
print("Done")
#---------------------------------------------------------------------------------------------#
uploading files to amazon s3 
'''BucketAlreadyOwnedByYou errors will only be returned outside of the US Standard region. 
Inside the US Standard region (i.e. when you don't specify a location constraint), attempting to recreate a bucket you #already own will succeed.'''
try:

    buck_name="ads-part2-assignment"

    S3_client = boto3.client('s3', Location, aws_access_key_id= accessKey, aws_secret_access_key=secretKey)
    
    if Location == 'us-east-1':
            S3_client.create_bucket(
                Bucket=buck_name,
            )
    else:
        S3_client.create_bucket(
                Bucket=buck_name,
                CreateBucketConfiguration={'LocationConstraint': Location},
            )
    logging.info("Connection is successful")
    S3_client.upload_file("ADSAssign1Part2.zip", buck_name, "ADSAssign1Part2.zip")
    S3_client.upload_file("Log_file2.log", buck_name, "Log_file2.log")
    logging.info("Files uploaded successfully")
except Exception as e:
    logging.error("Error uploading files to Amazon s3" + str(e))

