import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, datasets
from itertools import cycle, islice
from sklearn import preprocessing

np.random.seed(0)

data = datasets.make_moons(n_samples = 1000, noise = 0.11)
points = data[0]
plt.scatter(points[:,0], points[:,1])
plt.title('Data Points')
plt.show()

sc = cluster.SpectralClustering(n_clusters=2, affinity='nearest_neighbors', n_neighbors=5)
sc.fit(points)
y_sc = sc.fit_predict(points)

f5 = plt.figure()

plt.scatter(points[y_sc == 0, 0], points[y_sc == 0, 1], c = 'red')
plt.scatter(points[y_sc == 1, 0], points[y_sc == 1, 1], c = 'blue')
plt.scatter(points[y_sc == 2, 0], points[y_sc == 2, 1], c = 'black')
plt.scatter(points[y_sc == 3, 0], points[y_sc == 3, 1], c = 'cyan')

# X, y = datasets.make_moons(n_samples=1000, noise=0.1)
# scaler = preprocessing.StandardScaler()
# scaler.fit(X)
# X_Scaled = scaler.fit(X)
# dbscan = cluster.DBSCAN(eps=0.2)
# c = dbscan.fit_predict(X)
# plt.figure()
# print(c.max())