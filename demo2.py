
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, datasets
from itertools import cycle, islice


# In[2]:


np.random.seed(0)


# In[7]:


data = datasets.make_moons(n_samples = 1000, noise = 0.1)
points = data[0]
plt.scatter(points[:,0], points[:,1])
plt.title('Data Points')
plt.show()


# In[6]:


kmeans = cluster.KMeans(n_clusters=4)
kmeans.fit(points)

