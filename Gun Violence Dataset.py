#!/usr/bin/env python
# coding: utf-8

# In[122]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[123]:


import matplotlib.pyplot as plt
df = pd.read_csv('C:\\Users\\admir\\Desktop\\gun-violence-data_01-2013_03-2018.csv')


# In[124]:


df.head()


# In[125]:


df.info()
#Our dataset consists of 29 columns of up to 239677 rows of data. 


# In[126]:


df['state'].min() #Most popular country for gun violence of 2018


# In[127]:


df['state'].max() #Least popular country for gun violence of 2018


# In[128]:


df['date'].max() #Date with most reported acts of gun violence.


# In[129]:


plt.plot('n_guns_involved', 'n_injured', data=df)


# In[130]:


df.drop(columns=['participant_name','participant_relationship','participant_status'], axis=1)


# In[131]:


df.drop(columns=['notes', 'participant_age_group','incident_url_fields_missing','participant_relationship','sources','incident_url','source_url','congressional_district','location_description','longitude'], axis=1)


# In[132]:


df['n_guns_involved'].dropna(inplace=True)


# In[133]:


df.info()


# In[134]:


plt.figure(figsize=(20,20))
plt.scatter('n_killed', 'state', data=df)
plt.show()
#This plot tells us that Florida has had the highest number of Deaths related to gun violence in 2018. Let's examine it further.


# In[135]:


df2 = df[df['state']=='Florida']


# In[136]:


df2


# In[143]:


df2['city_or_county'].values.argmax()


# In[ ]:




