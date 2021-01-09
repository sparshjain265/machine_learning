from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import math

data = []
with open('PDBbind2015_refined-core.dat', 'r') as f:
    d = f.readline()
    d = f.readlines()
    for i in d:
        k = i.strip().split()
        data.append([float(i) for i in k]) 
data = np.array(data)
X = data[:, [2, 3, 4, 5, 6]]
Y = data.transpose()[0]
X_train , X_cv , Y_train , Y_cv = train_test_split(X , Y, random_state = 0)

train = [-math.inf]*3
valid = [-math.inf]*3
corr = [-math.inf]*3
deg = [0]*3
alpha = [0]*3

print("Training for different degrees in OLSR (1 - 4)")
train[0] = -math.inf
valid[0] = -math.inf
corr[0] = -math.inf
deg[0] = 0
for d in range(1, 5):
	poly = PolynomialFeatures (degree = d)
	poly.fit (X_train)
	X_train_poly = poly.transform(X_train)
	X_cv_poly = poly.transform(X_cv)
	
	OLSR = linear_model.LinearRegression(normalize = True)
	OLSR.fit(X_train_poly, Y_train)
	if(OLSR.score(X_cv_poly, Y_cv) > valid[0]):
		train[0] = OLSR.score(X_train_poly, Y_train)
		valid[0] = OLSR.score(X_cv_poly, Y_cv)
		corr[0] = np.corrcoef(OLSR.predict(X_cv_poly), Y_cv)[0, 1]
		deg[0] = d
print("Best OLSR Cross Validation in degree: " + str(deg[0]))
print("OLSR Training Accuracy: " + str(train[0]))
print("OLSR Cross Validation Accuracy: " + str(valid[0]))
print ("OLSR Correlation Coefficient: " + str(corr[0]))
print()

print("Training for different degrees in Ridge Regression (1 - 4)")
print("Alpha choosen from " + str([0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]))
train[1] = -math.inf
valid[1] = -math.inf
corr[1] = -math.inf
deg[1] = 0
alpha[1] = 0
for d in range(1, 5):
	poly = PolynomialFeatures (degree = d)
	poly.fit (X_train)
	X_train_poly = poly.transform(X_train)
	X_cv_poly = poly.transform(X_cv)
	for a in [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]:
		Ridge = linear_model.Ridge(normalize = True, alpha = a)
		Ridge.fit(X_train_poly, Y_train)
		if(Ridge.score(X_cv_poly, Y_cv) > valid[1]):
			train[1] = Ridge.score(X_train_poly, Y_train)
			valid[1] = Ridge.score(X_cv_poly, Y_cv)
			corr[1] = np.corrcoef(Ridge.predict(X_cv_poly), Y_cv)[0, 1]
			deg[1] = d
			alpha[1] = a
print("Best Ridge Cross Validation in degree: " + str(deg[1]) + " and alpha: " + str(alpha[1]))
print("Ridge Training Accuracy: " + str(train[1]))
print("Ridge Cross Validation Accuracy: " + str(valid[1]))
print ("Ridge Correlation Coefficient: " + str(corr[1]))
print()

print("Training for different degrees in LASSO Regression (1 - 4)")
print("Alpha choosen from " + str([0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]))
train[2] = -math.inf
valid[2] = -math.inf
corr[2] = -math.inf
deg[2] = 0
alpha[2] = 0
for d in range(1, 5):
	poly = PolynomialFeatures (degree = d)
	poly.fit (X_train)
	X_train_poly = poly.transform(X_train)
	X_cv_poly = poly.transform(X_cv)
	for a in [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]:
		Lasso = linear_model.Lasso(normalize = True, alpha = a, max_iter = 100000)
		Lasso.fit(X_train_poly, Y_train)
		if(Lasso.score(X_cv_poly, Y_cv) > valid[2]):
			train[2] = Lasso.score(X_train_poly, Y_train)
			valid[2] = Lasso.score(X_cv_poly, Y_cv)
			corr[2] = np.corrcoef(Lasso.predict(X_cv_poly), Y_cv)[0, 1]
			deg[2] = d
			alpha[2] = a
print("Best LASSO Cross Validation in degree: " + str(deg[2]) + " and alpha: " + str(alpha[2]))
print("LASSO Training Accuracy: " + str(train[2]))
print("LASSO Cross Validation Accuracy: " + str(valid[2]))
print ("LASSO Correlation Coefficient: " + str(corr[2]))
print()

max = np.argmax(valid)
model = ["OLSR", "Ridge", "LASSO"]
print("Best model: " + model[max] + " (Based on max Cross Validation data set accuracy)")
print("Degree: " + str(deg[max]))
print("Alpha: " + str(alpha[max]))
print("Training Accuracy: " + str(train[max]))
print("Cross Validation Accuracy: " + str(valid[max]))
print("Correlation Coefficient (On Cross Cross Validation Data Points): " + str(corr[max]))
print("Correlation Coefficient of given prediction with actual value: " + str(np.corrcoef(data[:, 1], data[:, 0])[0, 1]))

