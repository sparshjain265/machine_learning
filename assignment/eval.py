from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import math

data = []
fileName = input("Enter data file name: ")
with open('PDBbind2015_refined-core.dat', 'r') as f:
    d = f.readline()
    d = f.readlines()
    for i in d:
        k = i.strip().split()
        data.append([float(i) for i in k]) 
data = np.array(data)
X = data[:, [2, 3, 4, 5, 6]]
Y = data.transpose()[0]

degree = 2
alpha = 0.01

poly = PolynomialFeatures(degree = degree)
poly.fit(X)
X_poly = poly.transform(X)
reg = linear_model.Ridge(normalize = True, alpha = alpha)
reg.fit(X_poly, Y)
print("Training Accuracy: " + str(reg.score(X_poly, Y)))

data = []
with open(fileName, 'r') as f:
    d = f.readline()
    d = f.readlines()
    for i in d:
        k = i.strip().split()
        data.append([float(i) for i in k]) 
data = np.array(data)
X_test = data[:, [2, 3, 4, 5, 6]]
Y_test = data.transpose()[0]
X_poly_test = poly.transform(X)
print("Test Accuracy: " + str(reg.score(X_poly_test, Y_test)))
print("Correlation Coefficient of prediction from model: " + str(np.corrcoef(reg.predict(X_poly_test), Y_test)[0, 1]))
print("Correlation Coefficient of previous prediction: " + str(np.corrcoef(data[:, 1], data[:, 0])[0, 1]))