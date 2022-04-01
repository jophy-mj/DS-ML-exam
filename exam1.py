#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np
x=np.array([[1,2],[3,4]])
y=np.array([[3,2],[1,4]])

print("First matrix:\n",x)
print("Second matrix:\n",y)
print("Transpose of first matrix:\n",x.transpose())
print("Transpose of second matrix:\n",y.transpose())
m=np.dot(x,y)
print("Matrix multiplication:\n",m)


# In[14]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv("customer_data.csv")
data.head()


# In[21]:


data.info()


# In[36]:


xax=data.iloc[:,0]
yax=data.iloc[:,4]
plt.scatter(xax,yax)
plt.xlabel("Annual income(k$)")
plt.ylabel("Spending score")
plt.title("customer data")
plt.show()


# In[26]:


from sklearn.model_selection import train_test_split
x_train,y_train,x_test,y_test=train_test_split(x,y,test_size=0.1,random_state=0)
print(x_train)


# In[27]:


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
print(regressor.intercept_)


# In[28]:


print(regressor.coef_)


# In[31]:


x_axis=['age']
c=np.arange(x_axis)
Xax=np.arange(c)
x_axis=np.arange(Xax)
plt.scatter(x_axis)
plt.xlabel("annual income")
plt.ylabel("spending score")
plt.legend()
plt.show()


# In[37]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv("student_scores.csv")
data.head()


# In[38]:


data.info()


# In[40]:


xax=data.iloc[:,0]
yax=data.iloc[:,1]
plt.scatter(xax,yax)
plt.xlabel("Hours")
plt.ylabel("scores")
plt.title("student data")
plt.show()


# In[41]:


from sklearn.model_selection import train_test_split
x_train,y_train,x_test,y_test=train_test_split(x,y,test_size=0.1,random_state=0)
print(x_train)


# In[43]:


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
print(regressor.intercept_)


# In[44]:


print(regressor.coef_)


# In[51]:


x_axis=['hour']
c=np.arange(x_axis)
Xax=np.arange(c)
x_axis=np.arange(Xax)
plt.scatter(x_axis)
plt.xlabel("hours")
plt.ylabel("Test score")
plt.title("predicted value")
plt.legend()
plt.show()


# In[ ]:




