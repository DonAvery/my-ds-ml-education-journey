#!/usr/bin/env python
# coding: utf-8

# - 5.1 Intro / Session overview
# - 5.2 Saving and loading the model
# - 5.3 Web services: introduction to Flask
# - 5.4 Serving the churn model with Flask
# - 5.5 Python virtual environment: Pipenv
# - 5.6 Environment management: Docker
# - 5.7 Deployment to the cloud: AWS Elastic Beanstalk (optional)
# - 5.8 Summary
# - 5.9 Explore more
# - 5.10 Homework

# # 5.1 Intro / Session overview
# 
# In this session we talked about the earlier model we made in chapter 3 for churn prediction.
# This chapter containes the deployment of the model. If we want to use the model to predict new values without running the code, There's a way to do this. The way to use the model in different machines without running the code, is to deploy the model in a server (run the code and make the model). After deploying the code in a machine used as server we can make some endpoints (using api's) to connect from another machine to the server and predict values.
# 
# To deploy the model in a server there are some steps:
# 
# - After training the model save it, to use it for making predictions in future (session 02-pickle).
# - Make the API endpoints in order to request predictions. (session 03-flask-intro and 04-flask-deployment)
# - Some other server deployment options (sessions 5 to 9)
# 
# ![overview.png](attachment:overview.png)

# In[1]:


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

import urllib.request


# In[2]:


url = 'https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv'

filename = 'data-week-3.csv'

df = pd.read_csv(url)


# In[3]:


df.columns = df.columns.str.lower().str.replace(' ', '_')

categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)

for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ', '_')

df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
df.totalcharges = df.totalcharges.fillna(0)

df.churn = (df.churn == 'yes').astype(int)


# In[4]:


df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)


# In[5]:


numerical = ['tenure', 'monthlycharges', 'totalcharges']

categorical = [
    'gender',
    'seniorcitizen',
    'partner',
    'dependents',
    'phoneservice',
    'multiplelines',
    'internetservice',
    'onlinesecurity',
    'onlinebackup',
    'deviceprotection',
    'techsupport',
    'streamingtv',
    'streamingmovies',
    'contract',
    'paperlessbilling',
    'paymentmethod',
]


# In[6]:


def train(df_train, y_train, C=1.0):
    dicts = df_train[categorical + numerical].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X_train, y_train)
    
    return dv, model


# In[7]:


def predict(df, dv, model):
    dicts = df[categorical + numerical].to_dict(orient='records')

    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred


# In[8]:


C = 1.0
n_splits = 5


# In[9]:


kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

scores = []

for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = df_train.churn.values
    y_val = df_val.churn.values

    dv, model = train(df_train, y_train, C=C)
    y_pred = predict(df_val, dv, model)

    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)

print('C=%s %.3f +- %.3f' % (C, np.mean(scores), np.std(scores))) 


# In[10]:


scores


# In[12]:


dv, model = train(df_full_train, df_full_train.churn.values, C=1.0)
y_pred = predict(df_test, dv, model)

y_test = df_test.churn.values
auc = roc_auc_score(y_test, y_pred)
auc


# # 5.2 Saving and loading the model
# 
# In this session we'll cover the idea "How to use the model in future without training and evaluating the code"
# 
# - To save the model we made before there is an option using the pickle library:
#     - First install the library with the command pip install pickle-mixin if you don't have it.
#     - After training the model and being the model ready for prediction process use this code to save the model for later.
#     - `import pickle
#       with open('model.bin', 'wb') as f_out:
#          pickle.dump((dcit_vectorizer, model), f_out)
#       f_out.close() ## After opening any file it's nessecery to close it`
#          
#     - In the code above we'll making a binary file named model.bin and writing the dict_vectorizer for one hot encoding and model as array in it. (We will save it as binary in case it wouldn't be readable by humans)
#     - To be able to use the model in future without running the code, We need to open the binary file we saved before.
#     - `with open('mode.bin', 'rb') as f_in:  ## Note that never open a binary file you do not trust!
#          dict_vectorizer, model = pickle.load(f_in)
#        f_in.close()`
#     - With unpacking the model and the dict_vectorizer, We're able to again predict for new input values without training a new model by re-running the code.

# ### Saving the model

# In[13]:


import pickle


# In[14]:


output_file = f'model_C={C}.bin'
output_file


# `'wb'` is write binary, `'rb'` is read binary.

# In[16]:


with open(output_file, 'wb') as f_out:  
    pickle.dump((dv, model), f_out)


# ### Loading the model

# In[1]:


import pickle


# In[5]:


model_file = 'model_C=1.0.bin'


# In[6]:


with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)


# In[7]:


dv, model


# In[8]:


customer = {
    'gender': 'female',
    'seniorcitizen': 0,
    'partner': 'yes',
    'dependents': 'no',
    'phoneservice': 'no',
    'multiplelines': 'no_phone_service',
    'internetservice': 'dsl',
    'onlinesecurity': 'no',
    'onlinebackup': 'yes',
    'deviceprotection': 'no',
    'techsupport': 'no',
    'streamingtv': 'no',
    'streamingmovies': 'no',
    'contract': 'month-to-month',
    'paperlessbilling': 'yes',
    'paymentmethod': 'electronic_check',
    'tenure': 1,
    'monthlycharges': 29.85,
    'totalcharges': 29.85
}


# In[9]:


X = dv.transform([customer])


# In[12]:


model.predict_proba(X)[0, 1]


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




