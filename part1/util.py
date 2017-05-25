#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 08:27:57 2017

@author: pj
"""

import numpy as np
import pandas as pd

def init_weight_and_bias(M1, M2):
    W = np.random.randn(M1,M2) / np.sqrt(M1+M2)
    b = np.random.zeros(M2)
    
    return W.astype(np.float32), b.astype(np.float32)

def init_filter(shape, poolsz):
    w = np.random.randn(*shape) / np.sqrt(np.prod(shape[1:]) + shape[0]*np.prod(shape[2:]) /np.prod(poolsz))
    return w.astype(np.float32)
def relu(x):
    return x*(x>0)

def sigmoid(A):
    return 1/(1 + np.exp(-A))

def softmax(A):
    expA = np.exp(A)
    return expA/expA.sum(axis = 1, keepdims = True)

def sigmoid_cost(T, Y):
    return -(T.np.log(Y) + (1-T)*np.log(1-Y)).sum()

def cost(T,Y):
    return -(T*np.log(Y)).sum()

def cost2(T, Y):
    #same as function cost
    N = len(T)
    return - np.log(Y[np.arange(N), T]).sum()

def error_rate(targets, predictions):
    return np.mean( targets != predictions)

def y2indicator(y):
    N = len(y)
    K = len(set(y))
    ind = np.zeros((N, K))
    for i in xrange(N):
        ind[i, y[i]] = 1
    return ind

def getData(balance_ones = True):
    
    Y = []
    X = []
    
    first = True
    for line in open('fer2013.csv'):
        if first:
            first = False
        else:
            row = line.split(',')
            
            Y.append(int(row[0]))
            X.append([int[p] for p in row[1].split()])
            
    X, Y = np.array(X) /255.0 , np.array(Y)
    
    if balance_ones:
        X0, Y0 = X[Y!=1, :], Y[Y!=1]
        X1 = X[Y==1,:]
        X1 = np.repeat(X1, 9, axis = 0)
        X = np.vstack(X0, X1)
        Y = np.concatenate((Y0, [1]*len(X1)))
    return X, Y

def getImageData():
    X,Y = getData()
    N, D = X.shape
    d = int(np.sqrt(D))
    X = X.reshape(N,1, d,d)
    return X, Y

def getBinaryData():
    
    Y = []
    X = []
    
    first = True
    for line in open('fer2013.csv'):
        if first:
            first = False
        else:
            row = line.split(',')
            y = int(row[0])
            if y == 0 or y == 1:
            
                Y.append(int(row[0]))
                X.append([int[p] for p in row[1].split()])
            
    X, Y = np.array(X) /255.0 , np.array(Y)
    return X,Y



    