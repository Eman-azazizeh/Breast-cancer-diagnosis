#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import preprocessing
import seaborn as sns
import datetime as dt
import os
from sklearn.metrics import confusion_matrix


# In[5]:


data = pd.read_csv("data.csv")
data.head(10)


# In[6]:


data['diagnosis'] = data['diagnosis'].map({'M':1,'B':0})
data.head(10)


# In[7]:


data.drop('id',axis=1,inplace=True)
data.drop('Unnamed: 32',axis=1,inplace=True)
data.head(10)


# In[8]:


X=data.drop(['diagnosis'],axis=1).values
y=data['diagnosis'].values


# In[9]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=12)


# In[10]:


data.describe()


# In[11]:


datas = pd.DataFrame(preprocessing.scale(data.iloc[:,1:32]))
datas.columns = list(data.iloc[:,1:32].columns)
datas['diagnosis'] = data['diagnosis']

datas.diagnosis.value_counts().plot(kind='bar', alpha = 0.5, facecolor = 'b', figsize=(12,6))
plt.title("Diagnosis (M=1 , B=0)", fontsize = '18')
plt.ylabel("Total Number of Patients")
plt.grid(b=True)


# In[12]:


data.columns


# In[13]:


data_mean = data[['diagnosis','radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean', 'compactness_mean', 'concavity_mean','concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']]


# In[14]:


plt.figure(figsize=(14,14))
foo = sns.heatmap(data_mean.corr(), vmax=1, square=True, annot=True)


# In[16]:


from sklearn.ensemble import AdaBoostClassifier
adaboost = AdaBoostClassifier(n_estimators=100, base_estimator= None,learning_rate=1, random_state = 1)
adaboost.fit(X_train,y_train)


# In[20]:


y_pred = adaboost.predict(X_test)


# In[21]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
accuracy = float(cm.diagonal().sum())/len(y_test)
print("\nAccuracy Of AdaBoost For The Given Dataset : ", accuracy)


# In[28]:


from sklearn.metrics import accuracy_score,precision_score,recall_score
print("Accuracy:",accuracy_score(y_test, y_pred))
print("Precision:",precision_score(y_test, y_pred))
print("Recall:",recall_score(y_test, y_pred))


# In[ ]:




