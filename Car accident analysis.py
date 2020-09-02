#!/usr/bin/env python
# coding: utf-8

# In[88]:


from sklearn.utils import resample
import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')


# # Import Data

# In[68]:


df_car=pd.read_csv('https://s3.us.cloud-object-storage.appdomain.cloud/cf-courses-data/CognitiveClass/DP0701EN/version-2/Data-Collisions.csv')
df_car.head()


# In[69]:


df_car.shape


# # Data Cleaning and Preparation

# In[70]:


df_group_one = df_car[['SEVERITYCODE','WEATHER','ROADCOND','LIGHTCOND']]
df_group_one.head()


# In[71]:


df_group_one.dtypes


# In[72]:


df_group_one[["WEATHER", "ROADCOND",'LIGHTCOND']] = df_group_one[["WEATHER", "ROADCOND",'LIGHTCOND']].astype("category")


# In[40]:


df_group_one.dtypes


# In[73]:


df_group_one['WEATHERCAT']= df_group_one["WEATHER"].cat.codes
df_group_one['ROADCONDCAT']= df_group_one["ROADCOND"].cat.codes
df_group_one['LIGHTCONDCAT']= df_group_one["LIGHTCOND"].cat.codes
df_group_one.head()


# In[74]:


df_group_one.dtypes


# # Balance Data

# In[75]:


df_group_one['SEVERITYCODE'].value_counts()


# In[77]:


df_majority = df_group_one[df_group_one.SEVERITYCODE==1]
df_minority = df_group_one[df_group_one.SEVERITYCODE==2]
df_majority_downsampled = resample(df_majority, 
                                 replace=False,    # sample without replacement
                                 n_samples=58188,     # to match minority class
                                 random_state=78297) # reproducible results
df_downsampled = pd.concat([df_majority_downsampled, df_minority])

df_downsampled.SEVERITYCODE.value_counts()


# # Modeling

# ## define X and Y

# In[81]:


x = np.asarray(df_downsampled[['WEATHERCAT','ROADCONDCAT','LIGHTCONDCAT']])
x[0:5]


# In[82]:


y = np.asarray(df_downsampled['SEVERITYCODE'])
y[0:5]


# ## Data Normalization

# In[84]:


from sklearn import preprocessing
x = preprocessing.StandardScaler().fit(x).transform(x)
x[0:5]


# In[85]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=4)
print('Train set:',x_train.shape,y_train.shape)
print('Test set:',x_test.shape,y_test.shape)


# ## KNN

# In[126]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
Ks = 30
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))
ConfustionMx = [];
for n in range(1,Ks):
    
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = n).fit(x_train,y_train)
    yhat=neigh.predict(x_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)

    
    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

mean_acc

plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy ', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Nabors (K)')
plt.tight_layout()
plt.show()
print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1 )


# In[127]:


k =11
#Train Model and Predict  
neigh = KNeighborsClassifier(n_neighbors = k).fit(x_train,y_train)
neigh


# In[128]:


yhat = neigh.predict(x_test)
yhat[0:5]


# In[129]:


print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(x_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))


# ## Decision Tree

# In[130]:


from sklearn.tree import DecisionTreeClassifier
accidentTree = DecisionTreeClassifier(criterion="entropy", max_depth = 7)
accidentTree # it shows the default parameters


# In[131]:


accidentTree.fit(x_train,y_train)
predTree = accidentTree.predict(x_test)
print (predTree [0:10])
print (y_test [0:10])


# ## Logistic Regression

# In[140]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

LR = LogisticRegression(C=6, solver='liblinear').fit(x_train,y_train)
LR


# In[141]:


yhat = LR.predict(x_test)
yhat


# In[142]:


yhat_prob = LR.predict_proba(x_test)
yhat_prob


# # Evaluation

# KNN

# In[136]:


from sklearn.metrics import f1_score
from sklearn.metrics import log_loss# predicted y
yhat_knn = neigh.predict(x_test)

# jaccard
jaccard_knn = jaccard_similarity_score(y_test, yhat_knn)
print("KNN Jaccard index: ", jaccard_knn)

# f1_score
f1_score_knn = f1_score(y_test, yhat_knn, average='weighted')
print("KNN F1-score: ", f1_score_knn)


# Decision Tree

# In[138]:


# predicted y
yhat_dt = accidentTree.predict(x_test)

# jaccard
jaccard_dt = jaccard_similarity_score(y_test, yhat_dt)
print("DT Jaccard index: ", jaccard_dt)

# f1_score
f1_score_dt = f1_score(y_test, yhat_dt, average='weighted')
print("DT F1-score: ", f1_score_dt)


# Logistic Regression

# In[143]:


# predicted y
yhat_lg = LR.predict(x_test)
yhat_lg_prob = LR.predict_proba(x_test)

# jaccard
jaccard_lg = jaccard_similarity_score(y_test, yhat_lg)
print("LR Jaccard index: ", jaccard_lg)

# f1_score
f1_score_lg = f1_score(y_test, yhat_lg, average='weighted')
print("LR F1-score: ", f1_score_lg)

# logloss
logloss_lg = log_loss(y_test, yhat_lg_prob)
print("LR log loss: ", logloss_lg)


# # Report
# You should be able to report the accuracy of the built model using different evaluation metrics:

# | Algorithm          | Jaccard | F1-score | LogLoss |
# |--------------------|---------|----------|---------|
# | KNN                | 0.56302       | 0.547122        | NA      |
# | Decision Tree      | 0.56285       | 0.534773        | NA      |
# | LogisticRegression | 0.52435       | 0.509146        | 0.68563     |
