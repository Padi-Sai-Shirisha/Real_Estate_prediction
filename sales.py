#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')


# In[2]:


train = pd.read_csv('Train.csv')
test = pd.read_csv('Test.csv')


# In[3]:


train.head()


# In[4]:


test.head()


# In[5]:


train.shape


# In[6]:


test.shape


# In[7]:


print("Train Data Info:")
print(train.info())


# In[8]:


print("\nTest Data Info:")
print(test.info())


# In[9]:


print("\nTrain Data Statistics:")
print(train.describe())


# In[10]:


plt.figure(figsize=(8, 5))
sns.histplot(train['Item_Rating'], bins=20, kde=True)
plt.title('Distribution of Item Ratings')
plt.show()


# In[11]:


plt.figure(figsize=(8, 5))
sns.scatterplot(x='Item_Rating', y='Selling_Price', data=train)
plt.title('Relationship between Item Rating and Selling Price')
plt.show()


# In[12]:


train['Date'] = pd.to_datetime(train['Date'])
test['Date'] = pd.to_datetime(test['Date'])


# In[13]:


train['Month'] = train['Date'].dt.month
test['Month'] = test['Date'].dt.month
train['Year'] = train['Date'].dt.year
test['Year'] = test['Date'].dt.year


# In[14]:


plt.figure(figsize=(10, 6))
sns.lineplot(x='Year', y='Selling_Price', data=train, estimator='mean')
plt.title('Average Selling Price Over the Years')
plt.show()


# In[15]:


correlation_matrix = train[['Item_Rating', 'Selling_Price', 'Month', 'Year']].corr()

plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()


# In[16]:


plt.figure(figsize=(20, 16))
sns.barplot(x='Item_Category', y='Selling_Price', data=train)
plt.title('Average Selling Price by Item Category')
plt.xticks(rotation=45, ha='right')
plt.show()


# In[17]:


plt.figure(figsize=(20, 16))
sns.barplot(x='Subcategory_1', y='Selling_Price', data=train)
plt.title('Average Selling Price by Subcategory_1')
plt.xticks(rotation=45, ha='right')
plt.show()


# In[18]:


plt.figure(figsize=(20, 16))
sns.lineplot(x='Date', y='Selling_Price', data=train)
plt.title('Trend of Selling Price Over Time')
plt.show()


# In[19]:


plt.figure(figsize=(20, 16))
sns.boxplot(x='Item_Category', y='Item_Rating', data=train)
plt.title('Item Rating Distribution by Item Category')
plt.xticks(rotation=45, ha='right')
plt.show()


# In[20]:


X = train[['Item_Rating', 'Month', 'Year']]
y = train['Selling_Price']


# In[21]:


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# In[22]:


model = LinearRegression()
model.fit(X_train, y_train)


# In[23]:


predictions = model.predict(X_val)


# In[24]:


mse = mean_squared_error(y_val, predictions)
print("\nMean Squared Error on Validation Set:", mse)


# In[25]:


plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_val, y=predictions)
plt.title('Predicted vs Actual Selling Price')
plt.xlabel('Actual Selling Price')
plt.ylabel('Predicted Selling Price')
plt.show()


# In[ ]:

from sklearn import metrics
MAE = metrics.mean_absolute_error(y_val, predictions)
MSE = metrics.mean_squared_error(y_val, predictions)
RMSE = np.sqrt(MSE)

pd.DataFrame([MAE, MSE, RMSE], index=['MAE', 'MSE', 'RMSE'], columns=['Metrics'])

