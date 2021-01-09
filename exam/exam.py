from sklearn.datasets import make_moons
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

data = make_moons(n_samples=1500,noise=0.1)
X1 = data[:0]
X2 = data[:1]

X1Train, X1Test, X2Train, X2Test = train_test_split(X1, Y1, random_state = 0)

