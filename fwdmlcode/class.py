import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import datasets

def make_meshgrid(x, y, h=.02):
    
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

x1 = np.random.multivariate_normal([3,3],[[0.5,0],[0,0.2]],100)
x2 = np.random.multivariate_normal([3,6],[[0.5,0],[0,0.5]],100)

y1 = [1 for i in range(len(x1))]
y2 = [-1 for i in range(len(x2))]

x = np.concatenate((x1,x2))
y = np.concatenate((y1,y2))

#plt.scatter(x1[:,0],x1[:,1],color = 'red')
#plt.scatter(x2[:,0],x2[:,1],color = 'blue')
#plt.show()

clf = svm.SVC(C = 0.1,kernel = 'linear')
clf = clf.fit(x,y)
y_out = clf.predict(x)

#xx,yy = make_meshgrid(x[:,0],x[:,1],0.02)
#plot_contours(plt,clf,xx,yy,cmap = plt.cm.coolwarm,alpha = 0.8)

#model2
clf2 = svm.SVC(C = 0.1,kernel = 'rbf')
clf2 = clf2.fit(x,y)
y_out = clf2.predict(x)


#xx,yy = make_meshgrid(x[:,0],x[:,1],0.02)
#plot_contours(plt,clf2,xx,yy,cmap = plt.cm.coolwarm,alpha = 0.8)

#Iris dataset
iris = datasets.load_iris()
x_iris = iris.data[:,:2]
y_iris = iris.target
model = svm.SVC(C=1,kernel='linear')
model = model.fit(x_iris,y_iris)
y_predict = model.predict(x_iris)

#plt.scatter(x_iris[:,0],x_iris[:,1])
#plt.show()
xx,yy = make_meshgrid(x_iris[:,0],x_iris[:,1],0.02)
plot_contours(plt,model,xx,yy,cmap = plt.cm.coolwarm,alpha = 0.8)