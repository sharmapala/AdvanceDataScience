
# coding: utf-8

# In[13]:


#to extract the data using QuadAPI
import os # OS module provides function that allows you to interface with the operating system that Python is running on
import numpy as np #provides a high-performance multidimensional array object,
import pandas as pd
import pickle # Pickling is a way to convert a python object (list, dict, etc.) into a character stream
#serializing and de-serializing a Python object structure
import quandl #Quandl requires NumPy (v1.8 or above) and pandas (v0.14 or above) to work.
#Package for quandl API access
#Basic wrapper to return datasets from the Quandl website as Pandas dataframe objects with a timeseries index, or as a numpy array


# In[14]:


from datetime import datetime
py.init_notebook_mode(connected=True)


# In[15]:


def get_quandl_data(quandl_id):
    '''Download and cache Quandl dataseries'''
    #taking data series from cached path
    cache_path = '{}.pkl'.format(quandl_id).replace('/','-') # creat pkl file as quandl_id.pkl and replces / with -
    #cache_path = BCHARTS-KRAKENUSD.pkl
    try:
        f = open(cache_path, 'rb') #open the pkl file to read
        df = pickle.load(f) # load the data from file into df object 
        print('Loaded {} from cache'.format(quandl_id)) # loaded from the pkl file which was cached
    except (OSError, IOError) as e:
        print('Downloading {} from Quandl'.format(quandl_id))
        # getting data from quandl website using quandl module and API as pandas dataframe
        df = quandl.get(quandl_id, returns="pandas") 
        df.to_pickle(cache_path)# storing the data from df to pkl file 
        print('Cached {} at {}'.format(quandl_id, cache_path))
    return df



# In[18]:


#btc_usd_price_kraken = get_quandl_data('BCHARTS/KRAKENUSD') # fetch data from KRAKEN exchange
quandl_id = 'BCHARTS/{}USD'.format('COINBASE') 
#  historical Bitcoin exchange data for the Coinbase Bitcoin exchange:
df = get_quandl_data(quandl_id)
df.to_csv("bitcoin.csv")
df.head()
#btc_usd_price_kraken.head() #Python data visualization libraries such as Matplotlib,
#but I think Plotly is a great choice since it produces fully-interactive charts using D3.js


# In[10]:





# In[11]:


btc_usd_price_kraken.to_csv("dataset.csv")

