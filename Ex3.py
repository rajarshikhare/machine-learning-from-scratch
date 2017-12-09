
# coding: utf-8

# In[53]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
#get_ipython().magic('matplotlib inline')


# In[54]:


data = loadmat('ex3data1.mat')


# In[55]:


data


# In[56]:


X = data['X']
y = data['y']
y[y == 10] = 0


# In[57]:


X


# In[58]:


y


# In[59]:


y_n = np.zeros(shape = (5000, 10))
for i in range(0, X.shape[0]):
    y_n[i, y[i]] = 1


# In[60]:


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# In[61]:


plt.plot(sigmoid(np.array(range(-10, 10))))


# In[62]:


def cost(theta, X, y, lambda_r):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    m = len(X)
    term1 = np.multiply(-y, np.log(sigmoid(X*theta.T)))
    term2 = np.multiply(1-y, np.log(1-sigmoid(X*theta.T)))
    calc = np.sum(term1 - term2)
    reg = (lambda_r/(2 * m)) * np.sum(np.power(theta[0,1:], 2))
    return calc/m + reg


# In[63]:


def gradient(theta, X, y, lambda_r):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    m = len(X)
    error = sigmoid(X * theta.T) - y
    term = (lambda_r/m)*np.sum(theta[0:1:])
    grad = (X.T * error)/m + term
    return np.array(grad).ravel()


# In[147]:


initial_theta = np.zeros(X.shape[1])
learning_rate = 1


# In[148]:


from scipy.optimize import minimize
import scipy.optimize as opt
#result = opt.fmin_tnc(func=cost, x0=initial_theta, fprime=gradient, args=(X, y_n[:,i].reshape(y_n.shape[0], 1), learning_rate))
new_theta = np.zeros([X.shape[1],10])
for i in range(0, 10):
    #result = opt.fmin_tnc(func=cost, x0=initial_theta, fprime=gradient, args=(X, y_n[:,i].reshape(y_n.shape[0], 1), learning_rate))
    fmin = minimize(fun=cost, x0=initial_theta, args=(X, y_n[:,i].reshape(y_n.shape[0], 1), learning_rate), method='TNC', jac=gradient)
    new_theta[:, i] = fmin.x
    #new_theta[:, i] = result[0]


# In[149]:


def predict(new_theta, X):
    theta = np.matrix(new_theta)
    X = np.matrix(X)
    
    prediction = X * theta
    
    return prediction


# In[150]:


p = sigmoid(predict(new_theta, X))


# In[151]:


p[p < .5] = 0
p[p >= .5] = 1


# In[152]:


p


# In[153]:


ans = np.array(range(0, 5000))
count = 0
for i in range(0, X.shape[0]):
    for j in range(0, 10):
        if p[i, j] == 1:
            if y[i] == j:
                count = count + 1
print("Accuracy : ",(count*100/X.shape[0]))


# In[154]:


from skimage import io
img = 1 - io.imread('test5.png', as_grey=True)


# In[155]:


plt.imshow(img)


# In[156]:


image = np.matrix(img.ravel())
predict = image * new_theta
predict


# In[157]:


predict.argmax()

