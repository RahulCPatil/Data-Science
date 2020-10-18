#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


customers = pd.read_csv("Ecommerce Customers")


# In[3]:


customers.head()


# In[4]:


customers.describe()


# In[5]:


customers.info()


# In[7]:


sns.set_palette("GnBu_d")
sns.set_style('whitegrid')


# In[8]:


sns.jointplot(x='Time on Website',y='Yearly Amount Spent',data=customers)


# In[9]:


sns.jointplot(x='Time on App',y='Yearly Amount Spent',data=customers)


# In[10]:


sns.jointplot(x='Time on App',y='Length of Membership',kind = 'hex',data=customers)


# In[11]:


sns.pairplot(customers)


# In[13]:


sns.lmplot(x = 'Length of Membership',y='Yearly Amount Spent', data = customers)


# # Training and Testing Data

# In[15]:


customers.nunique()


# In[16]:


X = customers[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
y = customers['Yearly Amount Spent']


# In[17]:


from sklearn.model_selection import train_test_split


# In[18]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)


# # Training

# In[19]:


from sklearn.linear_model import LinearRegression


# In[20]:


lm = LinearRegression()


# In[21]:


lm.fit(X_train,y_train)


# In[22]:


print('Coefficients: \n', lm.coef_)


# # Predicting the Data

# In[23]:


predictions = lm.predict(X_test)


# In[24]:


plt.scatter(y_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')


# In[26]:


##Evaluating the Model


# In[27]:


from sklearn import metrics


# In[28]:


print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[29]:


sns.distplot((y_test-predictions),bins=50);


# In[30]:


coeffecients = pd.DataFrame(lm.coef_,X.columns)
coeffecients.columns = ['Coeffecient']
coeffecients


# Interpreting the coefficients:
# 
# Holding all other features fixed, a 1 unit increase in Avg. Session Length is associated with an increase of 25.98 total dollars spent.
# Holding all other features fixed, a 1 unit increase in Time on App is associated with an increase of 38.59 total dollars spent.
# Holding all other features fixed, a 1 unit increase in Time on Website is associated with an increase of 0.19 total dollars spent.
# Holding all other features fixed, a 1 unit increase in Length of Membership is associated with an increase of 61.27 total dollars spent.

# In[ ]:




