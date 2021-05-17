#!/usr/bin/env python
# coding: utf-8

# In[16]:


#Libraries
import math
import numpy as np
import pandas as pd
import pandas_datareader as web
from scipy.stats import norm
from scipy.special import ndtr as ndtr
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
np.random.seed(8)


# In[17]:


stock2 = input("Enter Ticker: ")


# In[18]:


HKDUSD = web.DataReader(stock2, data_source='yahoo', start='2001-07-15')
HKDUSD = HKDUSD.dropna()
HKDUSD


# In[19]:


HKDUSD['logReturn'] = np.log(HKDUSD['Close'].shift(-1)) - np.log(HKDUSD['Close'])


# In[20]:


#Calculations and Histogram
mu = HKDUSD['logReturn'].mean()
sigma = HKDUSD['logReturn'].std(ddof=1)

density = pd.DataFrame()
density['x'] = np.arange(HKDUSD['logReturn'].min()-0.01, HKDUSD['logReturn'].max()+0.01, 0.001)
density['pdf'] = norm.pdf(density['x'], mu, sigma)

HKDUSD['logReturn'].hist(bins=500, figsize=(18, 10))
plt.plot(density['x'], density['pdf'], color='red')
plt.show()


# In[21]:


#Value at Risk
VaR = norm.ppf(0.05, mu, sigma)
print('Yearly Value at Risk ', VaR)


# In[22]:


#Value at Risk
print('5% quantile ', norm.ppf(0.05, mu, sigma))
print('95% quantile', norm.ppf(0.95, mu, sigma))


# In[ ]:




