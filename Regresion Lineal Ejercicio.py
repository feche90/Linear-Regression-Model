#!/usr/bin/env python
# coding: utf-8

# In[153]:


import pandas as pd
import numpy as np


# In[154]:


df=open(r"C:\Users\feche\.spyder-py3\Curso Python\datasets\cereals\Cereal Data.txt","r")
df


# In[155]:


col_names=pd.read_excel(r"C:\Users\feche\.spyder-py3\Curso Python\datasets\cereals\Cereal data columns.xlsx")
col_names


# In[156]:


col_names_list=[]
col_names_list.append("Cereal")
col_names_list


# In[157]:


col_names_list2=col_names["Cereal"].tolist()


# In[158]:


for i in range(len(col_names_list2)):
    col_names_list.append(col_names_list2[i])

col_names_list


# In[159]:


counter=0
main_dic={}
for col in col_names_list:
    main_dic[col]=[]
main_dic


# In[160]:


for line in df:
    values=line.strip().split(" ")
    for i in range(len(col_names_list)):
        main_dic[col_names_list[i]].append(values[i])
    counter+=1

counter


# In[161]:


main_dic


# In[162]:


data=pd.DataFrame(main_dic)


# In[163]:


data.head()


# In[185]:


data.shape


# In[164]:


data[["Calories", "Protein", "Fat", "Sodium","Dietary Fiber", "Complex Carbohydrates", "Sugar ", "Potassium", "Vitamins & Minerals"]] = data[["Calories", "Protein", "Fat", "Sodium","Dietary Fiber", "Complex Carbohydrates", "Sugar ", "Potassium", "Vitamins & Minerals"]].apply(pd.to_numeric)
data.dtypes


# In[165]:


from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.svm import SVR


# In[166]:


feature_cols=["Protein", "Fat", "Sodium","Dietary Fiber", "Complex Carbohydrates", "Sugar ", "Potassium", "Vitamins & Minerals"]


# In[167]:


col_names=data.columns.values.tolist()
col_names


# In[168]:


X=data[feature_cols]
Y=data["Calories"]


# In[169]:


estimator=SVR(kernel="linear")
selector=RFE(estimator,8,step=1)
selector=selector.fit(X,Y)


# In[170]:


selector.support_


# In[171]:


selector.ranking_


# In[172]:


lm=LinearRegression()
lm.fit(X,Y)


# In[173]:


print(lm.intercept_)
print(lm.coef_)


# In[174]:


lm.score(X,Y)


# In[175]:


list(zip(feature_cols,lm.coef_))


# In[176]:


calories_pred=lm.predict(X)


# In[191]:


data["Calories_Pred"]=calories_pred
data.head(10)


# In[178]:


calories_mean=np.mean(data["Calories"])
calories_mean


# In[179]:


SSD=sum((data["Calories"]-data["Calories_Pred"])**2)
SSD


# In[180]:


RSE=np.sqrt(SSD/(len(data)-len(feature_cols)-1))
RSE


# In[184]:


error=RSE/calories_mean
error


# In[ ]:




