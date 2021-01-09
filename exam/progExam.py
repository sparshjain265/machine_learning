#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import make_moons
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


data = make_moons(n_samples=1500,noise=0.1)
data = data[0]
X = data[:,0:2]
print(X)


# In[3]:


XTrain, XTest= train_test_split(X)


# In[4]:


model = svm.SVC(kernel='polynomial', C = 100)

