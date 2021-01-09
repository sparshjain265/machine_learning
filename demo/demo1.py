import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import datasets

def make_meshgrid(x, y, h = 0.02):
	x_min, x_max = x.min() - 1, x.max() + 1
	y_min, y_max = y.min() - 1, y.max() + 1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
	return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
	z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
	z = z.reshape(xx.shape)
	out = ax.contourf(xx, yy, z, **params)
	return out

#Cluster 1
np.random.seed(0)
X11 = np.random.normal(3, 2, 100)
X12 = np.random.normal(3, 2, 100)
# X1 = np.random.multivariate_normal((3, 3), [[2, 0], [0, 2]], 300)
plt.scatter(X11, X12, marker = '.')
#mark the mean
plt.scatter(3, 3, s = 120, marker = '+', color = 'black')

# cluster 2
np.random.seed(0)
X21 = np.random.normal(3, 1, 100)
X22 = np.random.normal(6, 1, 100)

plt.scatter(X21, X22, marker = '.')
#mark the mean
plt.scatter(3, 6, s = 120, marker = '+', color = 'red')

plt.show()

Y1 = [1 for _ in range(len(X11))]
Y2 = [-1 for _ in range(len(X21))]

Y = Y1 + Y2
Y = np.asarray(Y)

X1 = np.column_stack((X11, X12))
X2 = np.column_stack((X21, X22))
X = np.append(X1, X2, axis = 0)
print("Shape of X: ", X.shape)
print("Shape of Y: ", Y.shape)

model = svm.SVC(kernel = 'linear', C = 0.1)
model = model.fit(X, Y)
y_out = model.predict(X)

xx, yy = make_meshgrid(X[:, 0], X[:, 1], 0.02)

plot_contours(plt, model, xx, yy, cmap = plt.cm.coolwarm, alpha = 0.8)