# -*- coding: utf-8 -*-
"""
Created on Sun May 21 14:34:07 2017

@author: pj
"""

import numpy as np
import matplotlib.pyplot as plt
import os

os.chdir('/home/pj/udemy/part1')

Nclass = 500

x1 = np.random.randn(Nclass, 2) + np.array([0, -2])
x2 = np.random.randn(Nclass, 2) + np.array([2, 2])
x3 = np.random.randn(Nclass, 2) + np.array([-2, 2])
X = np.vstack([x1,x2,x3])

Y = np.array([0]*Nclass + [1]*Nclass + [2]*Nclass)

D = 2
K = 3
M = 3

plt.scatter(X[:,0], X[:,1], c = Y, s= 100, alpha = 0.5)

W1 = np.random.randn(D, M)
b1 = np.random.randn(M)
W2 = np.random.randn(M, K)
b2 = np.random.randn(K)

def forward(X, W1, b1, W2, b2):
    #sigmod
    z = 1/ (1 + np.exp(-X.dot(W1) - b1))
    A = z.dot(W2) + b2
    expA = np.exp(A)
    Y = expA/expA.sum(axis = 1, keepdims = True)
    return Y
    
    
def  classification_rate(Y, P):
    n_correct = 0
    n_total = 0
    for i in xrange(len(Y)):
        n_total += 1
        if Y[i] == P[i]:
            n_correct += 1           
    return float(n_correct)/n_total
    
P_Y_given_X = forward(X, W1, b1, W2, b2)

P = np.argmax(P_Y_given_X, axis = 1)

print classification_rate(Y, P)

assert(len(P)==len(Y))
