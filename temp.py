
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


data = pd.read_csv('ex1data1.txt', names=['A', 'B'])
data.insert(0, 'ones', 1)


# In[3]:


plt.plot(data.iloc[:,1], data.iloc[:,2], 'o')
plt.show()


# In[4]:


X = data.iloc[:,[0, 1]]
y = data.iloc[:,2]


# In[ ]:


X.head()


# In[ ]:


y.head()


# In[ ]:


X = np.matrix(X.values)
y = np.matrix(y.values).T


# In[ ]:


def costFunction(X, y, theta):
    return np.sum(np.power(((X*theta.T) - y), 2))/(2*len(X))


# In[ ]:


def gradienDescent(X, y, theta, iter, alpha):
    m = len(X)
    cost = list([])
    for i in range(1, iter):
        theta1 = theta[0,0] - (alpha/m) * np.sum(np.multiply((X*theta.T - y), X[:,0]))
        theta2 = theta[0,1] - (alpha/m) * np.sum(np.multiply((X*theta.T - y), X[:,1]))
        theta = np.matrix([theta1, theta2])
        cost.append(costFunction(X, y, theta))
    return theta, cost
    


# In[ ]:


initial_theta = np.matrix([0, 1])
iter = 1000
alpha = .01


# In[ ]:


theta, cost = gradienDescent(X, y, initial_theta, iter, alpha)
plt.plot(cost)
plt.show()


# In[ ]:


prediction = X * theta.T


# In[ ]:


theta


# In[ ]:


cost.pop()


# In[ ]:


plt.plot(X, prediction)
plt.plot(X, y, 'o')
plt.show()

