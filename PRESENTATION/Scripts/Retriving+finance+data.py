
# coding: utf-8

# In[63]:


import json
from urllib import *
from bs4 import BeautifulSoup
import requests


# In[66]:


url = 'https://www.kaggle.com/philmohun/cryptocurrency-financial-data'
html = urllib.request.urlopen('https://www.kaggle.com/mczielinski/bitcoin-historical-data/data').read()
#page = requests.get("https://www.kaggle.com/philmohun/cryptocurrency-financial-data")
#page
#data = json.load(urllib.request.urlopen(url).read())

html


# In[69]:


#page.content
soup = BeautifulSoup(html, "html.parser")


# In[18]:


#soup =BeautifulSoup(page.content, 'html.parser')


# In[71]:


print(soup)


# In[60]:


result = soup.find_all('table', attrs={'class': 'table table-bordered table-striped sheet-table'})


# In[61]:


len(result)


# In[62]:


result[0:3]


# In[14]:


first_col = result[0]
print(first_col)


# In[15]:


headline = first_col.find('h2', attrs={'class':'media-heading'}).find('a').text
print(headline)  #printing headline


# In[16]:


datetime = first_col.find('span', attrs={'class':'date'}).text
print(datetime) # fetches date and time


# In[17]:


author =first_col.find('a', attrs={'class':'author'}).text
print(author)


# In[18]:


content = first_col.find('div', attrs={'class':'content'}).find('p').text
print(content)


# In[19]:


record =[]
for res in result:
    headline = res.find('h2', attrs={'class':'media-heading'}).find('a').text
    datetime = res.find('span', attrs={'class':'date'}).text
    author =res.find('a', attrs={'class':'author'}).text
    content = res.find('div', attrs={'class':'content'}).find('p').text
    record.append((datetime, author,headline,content))


# In[20]:


len(record)


# In[21]:


record[0:3]


# In[27]:


import pandas as pd
df = pd.DataFrame(record, columns=['Date/Time', 'Author', 'Headline', 'About'])


# In[26]:


del df['Date/Time']
df


# In[89]:


df.to_csv('Web_scrapping.csv')

