#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import accuracy_score,confusion_matrix,roc_curve,roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[2]:


data = pd.read_csv('heart_disease.csv')


# In[3]:


data.head()


# In[4]:


data.info()


# In[8]:


data= data.drop(['Unnamed: 0'],axis=1)


# In[9]:


data.info()


# In[10]:


data.describe()


# In[12]:


data.shape


# In[13]:


data.isnull().sum()


# In[15]:


plt.figure(figsize=(20,15),facecolor ='yellow')
plotnumber =1
for column in data:
    if plotnumber <=14:
        ax= plt.subplot(4,4,plotnumber)
        sns.distplot(data[column])
        plt.xlabel(column,fontsize=20)
    plotnumber +=1
plt.show()


# In[16]:


df_feature = data.drop('target',axis=1)


# In[17]:


plt.figure(figsize=(20,25))
graph =1
for column in df_feature:
    if graph <=14:
        plt.subplot(4,4,graph)
        ax=sns.boxplot(data=df_feature[column])
        plt.xlabel(column,fontsize=15)
    graph +=1
plt.show()


# In[18]:


q1 = data.quantile(0.25)
q3 = data.quantile(0.75)
iqr = q3-q1


# In[19]:


trestbps_high = (q3.trestbps+ (1.5*iqr.trestbps))
print(trestbps_high)
index =np.where(data['trestbps']>trestbps_high)
data = data.drop(data.index[index])
print(data.shape)
data.reset_index()


# In[20]:


chol_high = (q3.chol+ (1.5*iqr.chol))
print(chol_high)
index =np.where(data['chol']>chol_high)
data = data.drop(data.index[index])
print(data.shape)
data.reset_index()


# In[21]:


thalach_low = (q3.thalach-(1.5*iqr.thalach))
print(thalach_low)
index =np.where(data['thalach']<thalach_low)
data = data.drop(data.index[index])
print(data.shape)
data.reset_index()


# In[22]:


old_peak_high = (q3.oldpeak+(1.5*iqr.oldpeak))
print(old_peak_high)
index =np.where(data['oldpeak']>old_peak_high)
data = data.drop(data.index[index])
print(data.shape)
data.reset_index()


# In[23]:


ca_high = (q3.ca+(1.5*iqr.ca))
print(ca_high)
index =np.where(data['ca']>ca_high)
data = data.drop(data.index[index])
print(data.shape)
data.reset_index()


# In[24]:


thal_low = (q3.thal-(1.5*iqr.thal))
print(thal_low)
index =np.where(data['thal']<thal_low)
data = data.drop(data.index[index])
print(data.shape)
data.reset_index()


# In[25]:


data= data.drop(['sex'],axis=1)
data=data.drop(['fbs'],axis=1)
data= data.drop(['exang'],axis=1)


# In[26]:


data.shape


# In[27]:


plt.figure(figsize =(20,25))
plotnumber = 1
for column in data:
    if plotnumber<=11:
        ax = plt.subplot(4,4,plotnumber)
        sns.distplot(data[column])
        plt.xlabel(column,fontsize=20)
    plotnumber +=1
plt.show()


# In[28]:


x=data.drop(columns =['target'])
y=data['target']


# In[30]:


plt.figure(figsize=(15,20))
plotnumber = 1

for column in x:
    if plotnumber <=11:
        ax= plt.subplot(4,4,plotnumber)
        sns.stripplot(x=y,y=x[column],hue=y)
    plotnumber +=1
plt.show()


# In[31]:


scaler= StandardScaler()
x_scaled = scaler.fit_transform(x)


# In[32]:


x_scaled.shape[1]


# In[33]:


vif = pd.DataFrame()
vif['vif']=[variance_inflation_factor(x_scaled,i)for i in range(x_scaled.shape[1])]
vif['Features']=x.columns


vif


# In[34]:


x_train,x_test,y_train,y_test=train_test_split(x_scaled,y,test_size=0.25,random_state =355)


# In[35]:


log_reg = LogisticRegression()
log_reg.fit(x_train,y_train)


# In[36]:


y_pred= log_reg.predict(x_test)


# In[37]:


y_pred


# In[38]:


log_reg.predict_proba(x_test)


# In[39]:


conf_mat=confusion_matrix(y_test,y_pred)
conf_mat


# In[40]:


accuracy = accuracy_score(y_test,y_pred)
accuracy


# In[41]:


from sklearn.metrics import classification_report


# In[42]:


print(classification_report(y_test,y_pred))


# In[ ]:




