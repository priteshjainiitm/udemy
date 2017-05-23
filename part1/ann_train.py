#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 21:09:43 2017

@author: pj
"""

import numpy as np
import matplotlib.pyplot as plt
from process import get_data
from sklearn.utils import shuffle

def y2indicator(Y, K):
    N = len(Y)
    ind = np.zeros((N, K))
    for i in xrange(N):
        ind[i, Y[i]] = 1
    return ind
    
X, Y = get_data()
X, Y = shuffle(X, Y)

Y = Y.astype(np.int32)

M = 5
D = X.shape[1]
K = len(set(Y))

Xtrain = X[:-100]
Ytrain = Y[:-100]
Ytrain_ind = y2indicator(Ytrain, K)

Xtest = X[-100:]
Ytest = Y[-100:]
Ytest_ind = y2indicator(Ytest, K)

W1 = np.random.randn(D, M)
b1 = np.random.randn(M)
W2 = np.random.randn(M,K)
b2 = np.random.randn(K)

def softmax(a):
    expA = np.exp(a)
    return expA/expA.sum(axis = 1, keepdims = True)

def forward(X, W1, b1, W2, b2):
    z = np.tanh(X.dot(W1)+b1)
    return softmax(z.dot(W2)+b2), z

#P_Y_given_X = forward(X, W1, b1, W2, b2)
#predictions = np.argmax(P_Y_given_X, axis = 1)

def predict(P_Y_given_X):
    return np.argmax(P_Y_given_X, axis = 1)
    
def classification_rate(Y, P):
    return np.mean(Y==P)
    
def cross_entropy(T, pY):
    return (-np.mean(T*np.log(pY)))
    
train_costs = []
test_costs = []
learning_rate = 0.001

for i in xrange(10000):
    pYtrain, Ztrain = forward(Xtrain, W1, b1, W2, b2)
    pYtest, Ztest = forward(Xtest, W1, b1, W2, b2)
    
    ctrain = cross_entropy(Ytrain_ind, pYtrain)
    ctest = cross_entropy(Ytest_ind, pYtest)
    
    W2 -= learning_rate*Ztrain.T.dot(pYtrain - Ytrain_ind)
    b2 -= learning_rate*(pYtrain - Ytrain_ind).sum()  
    dz = (pYtrain - Ytrain_ind).dot(W2.T)*(1- Ztrain*Ztrain)#tanh
    W1 -= learning_rate*Xtrain.T.dot(dz)
    b1 -= learning_rate*dz.sum(axis = 0)
    
    if i%1000 == 0:
        print i, ctrain, ctest
        
print 'train acc.', classification_rate(Ytrain,predict(pYtrain))
print 'test acc.', classification_rate(Ytest, predict(pYtest))

legend1, = plt.plot(train_costs, label = 'Train Cost')
legend2, = plt.plot(test_costs, label = 'Test Cost')

plt.legend([legend1,legend2])
plt.show()