#!/usr/bin/env python
# coding: utf-8

# In[2]:


from pandas_datareader import data,wb
import pandas as pd
import numpy as np
import datetime
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


start = datetime.datetime(2006,1,1)
end = datetime.datetime(2016,1,1)


# In[5]:


BAC = data.DataReader("BAC",'yahoo',start,end)
C = data.DataReader("C",'yahoo',start,end)
GS = data.DataReader("GS",'yahoo',start,end)
JPM=data.DataReader("JPM",'yahoo',start,end)
MS=data.DataReader("MS",'yahoo',start,end)
WFC=data.DataReader("WFC",'yahoo',start,end)


# In[8]:


df = data.DataReader(['BAC','C','GS','JPM','MS','WFC'],'yahoo',start,end)


# In[9]:


tickers = ['BAC','C','GS','JPM','MS','WFC']


# In[13]:


bank_stocks = pd.concat([BAC,C,GS,JPM,MS,WFC],axis =1,keys=tickers)


# In[14]:


bank_stocks.columns.names = ['Bank Ticker', 'Stock Info']


# In[15]:


bank_stocks.head()


# In[16]:


bank_stocks.xs(key = 'Close', axis = 1,level = 'Stock Info').max()


# In[17]:


returns = pd.DataFrame()


# In[18]:


for tick in tickers:
    returns[tick+' Return'] = bank_stocks[tick]['Close'].pct_change()
returns.head()


# In[19]:


import seaborn as sns 
sns.pairplot(returns [1:])


# In[20]:


returns.idxmin()


# In[21]:


returns.idxmax()


# In[22]:


returns.std()


# In[23]:


returns.ix['2015-01-01':'2015-12-01'].std()


# In[43]:


sns.distplot(returns.ix['2015-01-01':'2015-12-01']['MS Return'],color = 'green', bins = 100)


# In[44]:


sns.distplot(returns.ix['2008-01-01':'2008-12-01']['C Return'],color = 'red', bins = 100)


# In[45]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')

# Optional Plotly Method Imports
import plotly
import cufflinks as cf
cf.go_offline()


# In[46]:


for tick in tickers:
    bank_stocks[tick]['Close'].plot(figsize=(12,4),label=tick)
plt.legend()


# In[47]:


bank_stocks.xs(key='Close',axis=1,level='Stock Info').plot()


# In[48]:


bank_stocks.xs(key='Close',axis=1,level='Stock Info').iplot()


# In[49]:


plt.figure(figsize=(12,6))
BAC['Close'].ix['2008-01-01':'2009-01-01'].rolling(window=30).mean().plot(label='30 Day Avg')
BAC['Close'].ix['2008-01-01':'2009-01-01'].plot(label='BAC CLOSE')
plt.legend()


# In[50]:


sns.heatmap(bank_stocks.xs(key='Close',axis=1,level='Stock Info').corr(),annot=True)


# In[51]:


sns.clustermap(bank_stocks.xs(key='Close',axis=1,level='Stock Info').corr(),annot=True)


# In[52]:


close_corr = bank_stocks.xs(key='Close',axis=1,level='Stock Info').corr()
close_corr.iplot(kind='heatmap',colorscale='rdylbu')


# In[ ]:




