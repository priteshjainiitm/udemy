import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

Nclass = 500

x1 = np.random.randn(Nclass, 2) + np.array([0, -2])
x2 = np.random.randn(Nclass, 2) + np.array([2, 2])
x3 = np.random.randn(Nclass, 2) + np.array([-2, 2])
X = np.vstack([x1,x2,x3])

Y = np.array([0]*Nclass + [1]*Nclass + [2]*Nclass)

plt.scatter(X[:,0], X[:,1], c = Y, s= 100, alpha = 0.5)

D = 2
K = 3
M = 3
N = len(Y)
T = np.zeros((N,K))
for i in xrange(N):
    T[i, Y[i]] = 1