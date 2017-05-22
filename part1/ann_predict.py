# -*- coding: utf-8 -*-
"""
Created on Sun May 21 16:45:27 2017

@author: pj
"""

import numpy as np
from process import get_data

X, Y = get_data()

M = 5
D = X.shape[1]
K = len(set(Y))

W1 = np.random.randn(D, M)
b1 = np.random.randn(M)
W2 = np.random.randn(M, K)
b2 = np.random.randn(K)

def softmax(a):
    expA = np.exp(a)
    return expA/expA.sum(axis = 1, keepdims = True)

def forward(X, W1, b1, W2, b2):
    z = np.tanh(X.dot(W1)+b1)
    return softmax(z.dot(W2)+b2)

P_Y_given_X = forward(X, W1, b1, W2, b2)
predictions = np.argmax(P_Y_given_X, axis = 1)

def classification_rate(Y, P):
    np.mean(Y == P)

print 'score', classification_rate(Y, predictions)    