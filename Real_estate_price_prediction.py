#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df=pd.read_csv('Real estate.csv')


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


sns.pairplot(df)
plt.show()


# In[7]:


sns.distplot(df['Y house price of unit area'], color='green')
plt.show()


# In[8]:


x=df.drop(['Y house price of unit area', 'No'], axis=1)
y=df['Y house price of unit area']


# In[9]:


x.head(10)


# In[10]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=50)


# In[11]:


x_train.head(10)


# In[12]:


from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x_train, y_train)


# In[13]:


pd.DataFrame(reg.coef_, x.columns, columns=['coefficient'])


# In[14]:


reg.intercept_


# In[15]:


df['Y house price of unit area'].value_counts()


# In[16]:


y_predict = reg.predict(x_test)
pd.DataFrame({'Test':y_test, 'Prediction':y_predict}).head(10)



# In[18]:


df['Y house price of unit area'].mean()


# In[19]:


residuals = y_test-y_predict
sns.scatterplot(x=y_test, y=y_predict, color='olive')
plt.xlabel('y_test')
plt.ylabel('y_predict')
plt.show()


# In[20]:


sns.distplot(residuals, color='r', hist=False)


# In[21]:


sns.scatterplot(x=y_test, y=residuals)
plt.axhline(y=0, color='r', ls='--')


# In[22]:


plt.scatter(reg.predict(x_train), reg.predict(x_train) - y_train, color = "green", s = 35, edgecolor='black', label = 'Train data')
plt.scatter(reg.predict(x_test), reg.predict(x_test) - y_test, color = "red", s = 35, edgecolor='black', label = 'Test data')
plt.hlines(y = 0, xmin = 0, xmax = 50, colors='#5e03fc', linewidth = 2)
plt.legend(loc = 'upper right')
plt.title("Residual errors")
plt.show()


from sklearn import metrics
MAE = metrics.mean_absolute_error(y_predict, y_test)
MSE = metrics.mean_squared_error(y_predict, y_test)
RMSE = np.sqrt(MSE)

pd.DataFrame([MAE, MSE, RMSE], index=['MAE', 'MSE', 'RMSE'], columns=['Metrics'])
