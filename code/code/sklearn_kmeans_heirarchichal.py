from sklearn.datasets import make_moons
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering as ac
import scipy.cluster.hierarchy as sch
from sklearn.cluster import Birch 
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering

import numpy as np
import matplotlib.pyplot as plt

data = make_moons(n_samples=1000,noise=0.1)

points = data[0]
plt.scatter(points[:,0],points[:,1])
plt.title('Data points')
plt.show()

#creating the kmeans object
kmeans = KMeans(n_clusters=4)

#fit the data points to the k means algorithm
kmeans.fit(points)

print(kmeans.cluster_centers_)
y_kmeans = kmeans.fit_predict(points)
f1 = plt.figure()
plt.title('K-means clustering')
plt.scatter(points[y_kmeans==0,0],points[y_kmeans==0,1],c='red')
plt.scatter(points[y_kmeans==1,0],points[y_kmeans==1,1],c='blue')
plt.scatter(points[y_kmeans==2,0],points[y_kmeans==2,1],c='black')
plt.scatter(points[y_kmeans==3,0],points[y_kmeans==3,1],c='cyan')
plt.show()


#create dendogram
#dendogram = sch.dendrogram(sch.linkage(points,method='ward'))
hc = ac(n_clusters=2,affinity='euclidean',linkage='ward')
y_hc = hc.fit_predict(points)
f2 = plt.figure()

plt.scatter(points[y_hc==0,0],points[y_hc==0,1],c='red')
plt.scatter(points[y_hc==1,0],points[y_hc==1,1],c='blue')
plt.scatter(points[y_hc==2,0],points[y_hc==2,1],c='black')
plt.scatter(points[y_hc==3,0],points[y_hc==3,1],c='cyan')
plt.title('Heirarchical Clustering')
plt.show()


#Birch clustering
bir = Birch(n_clusters = 2,threshold = 0.8,branching_factor = 200)
bir.fit(points)
y_bir = bir.fit_predict(points)

f3 = plt.figure()

plt.scatter(points[y_bir==0,0],points[y_bir==0,1],c='red')
plt.scatter(points[y_bir==1,0],points[y_bir==1,1],c='blue')
plt.scatter(points[y_bir==2,0],points[y_bir==2,1],c='black')
plt.scatter(points[y_bir==3,0],points[y_bir==3,1],c='cyan')
plt.title('Birch Clustering')
plt.show()

#DBSCAN
dbs = DBSCAN(eps = 0.1,min_samples=5)
dbs.fit(points)
y_dbs = dbs.fit_predict(points)

f4 = plt.figure()

plt.scatter(points[y_dbs==0,0],points[y_dbs==0,1],c='red')
plt.scatter(points[y_dbs==1,0],points[y_dbs==1,1],c='blue')
plt.scatter(points[y_dbs==2,0],points[y_dbs==2,1],c='black')
plt.scatter(points[y_dbs==3,0],points[y_dbs==3,1],c='cyan')
plt.title('DBSCAN Clustering')
plt.show()

#Spectral clustering
sc = SpectralClustering(n_clusters=2,affinity='nearest_neighbors',n_neighbors=10)
sc.fit(points)
y_sc = sc.fit_predict(points)

f5 = plt.figure()

plt.scatter(points[y_sc==0,0],points[y_sc==0,1],c='red')
plt.scatter(points[y_sc==1,0],points[y_sc==1,1],c='blue')
plt.scatter(points[y_sc==2,0],points[y_sc==2,1],c='black')
plt.scatter(points[y_sc==3,0],points[y_sc==3,1],c='cyan')
plt.title('Spectral Clustering')
plt.show()