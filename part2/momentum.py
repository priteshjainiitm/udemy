#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 29 21:03:47 2017

@author: pj
"""

import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

from util import get_normalized_data, cost, error_rate, y2indicator
from mlp import forward, derivative_b1, derivative_b2, derivative_w1, derivative_w2


def main():
    max_iter = 30
    print_period = 10
    
    X,Y = get_normalized_data()
    lr = 0.00004
    reg = 0.01
    
    Xtrain = X[:-1000, :]
    Ytrain = Y[:-1000]
    Xtest = X[-1000:, :]
    Ytest = Y[-1000:]
    Ytrain_ind = y2indicator(Ytrain)
    Ytest_ind = y2indicator(Ytest)
    
    N,D = Xtrain.shape
    batch_sz = 500
    n_batches = N/ batch_sz
    
    M = 300
    K = 10
    W1 = np.random.randn(D, M)/28
    b1 = np.zeros(M)
    W2 = np.random.randn(M, K)/ np.sqrt(M)
    b2 = np.zeros(K)
    
    #1 batch
    LL_batch = []
    CR_batch = []
    
    for i in xrange(max_iter):
        for j in xrange(n_batches):
            Xbatch = Xtrain[j*batch_sz:(j*batch_sz + batch_sz),]
            Ybatch  = Ytrain[j*batch_sz:(j*batch_sz+batch_sz)]
            pYbatch, Z = forward(Xbatch, W1, b1, W2, b2)
            
            #updates
            W2 -= lr*(derivative_w2(Z, Ybatch, pYbatch) + reg*W2)
            b2 -= lr*(derivative_b2(Ybatch, pYbatch) + reg*b2)
            W1 -= lr*(derivative_w1(Xbatch, Z, Ybatch, pYbatch, W2) + reg*W1)
            b1 -= lr*(derivative_b1(Z, Ybatch, pYbatch, W2) + reg*b1)
            
            if j% print_period ==0:
                pY, _ = forward(Xtest, W1, b1, W2, b2)
                ll = cost(pY, Ytest_ind)
                LL_batch.append(ll)
                print "Cost at iteration i = %d, j = %d : %.6f" %(i, j,ll)
                err = error_rate(pY, Ytest)
                CR_batch.append(err)
                print "Error rate:", err
                
    pY, _ = forward(Xtest, W1, b1, W2, b2)
    err = error_rate(pY, Ytest)
    
    #batch with momentum
    
    W1 = np.random.randn(D, M)/28
    b1 = np.zeros(M)
    W2 = np.random.randn(M, K)/ np.sqrt(M)
    b2 = np.zeros(K)
    

    LL_momentum = []
    CR_momentum = []
    mu = 0.5
    dW1 = 0
    db1 = 0
    dW2 = 0
    db2 = 0
    for i in xrange(max_iter):
        for j in xrange(n_batches):
            Xbatch = Xtrain[j*batch_sz:(j*batch_sz + batch_sz),]
            Ybatch  = Ytrain[j*batch_sz:(j*batch_sz+batch_sz)]
            pYbatch, Z = forward(Xbatch, W1, b1, W2, b2)
            
            #updates
            dW2 = dW2*mu - lr*(derivative_w2(Z, Ybatch, pYbatch) + reg*W2)
            W2 += dW2
            db2 = db2*mu - lr*(derivative_b2(Ybatch, pYbatch) + reg*b2)
            b2 += db2
            dW1 = mu*dW1 - lr*(derivative_w1(Xbatch, Z, Ybatch, pYbatch, W2) + reg*W1)
            W1 += dW1
            db1 = mu*db1 - lr*(derivative_b1(Z, Ybatch, pYbatch, W2) + reg*b1)
            b1 += db1
            
            
            
            if j% print_period ==0:
                pY, _ = forward(Xtest, W1, b1, W2, b2)
                ll = cost(pY, Ytest_ind)
                LL_momentum.append(ll)
                print "Cost at iteration i = %d, j = %d : %.6f" %(i, j,ll)
                err = error_rate(pY, Ytest)
                CR_momentum.append(err)
                print "Error rate:", err
                
    pY, _ = forward(Xtest, W1, b1, W2, b2)
    err = error_rate(pY, Ytest)
    
    #batch with Nesterov
    
    W1 = np.random.randn(D, M)/28
    b1 = np.zeros(M)
    W2 = np.random.randn(M, K)/ np.sqrt(M)
    b2 = np.zeros(K)
    

    LL_nesterov = []
    CR_nesterov = []
    mu = 0.5
    dW1 = 0
    db1 = 0
    dW2 = 0
    db2 = 0
    for i in xrange(max_iter):
        for j in xrange(n_batches):
            Xbatch = Xtrain[j*batch_sz:(j*batch_sz + batch_sz),]
            Ybatch  = Ytrain[j*batch_sz:(j*batch_sz+batch_sz)]
            pYbatch, Z = forward(Xbatch, W1, b1, W2, b2)
            
            #updates
            dW2 = dW2*mu*mu - (1+mu)*lr*(derivative_w2(Z, Ybatch, pYbatch) + reg*W2)
            W2 += dW2
            db2 = db2*mu*mu - (1+mu)*lr*(derivative_b2(Ybatch, pYbatch) + reg*b2)
            b2 += db2
            dW1 = mu*mu*dW1 - (1+mu)*lr*(derivative_w1(Xbatch, Z, Ybatch, pYbatch, W2) + reg*W1)
            W1 += dW1
            db1 = mu*mu*db1 - (1+mu)*lr*(derivative_b1(Z, Ybatch, pYbatch, W2) + reg*b1)
            b1 += db1
            
            
            
            if j% print_period ==0:
                pY, _ = forward(Xtest, W1, b1, W2, b2)
                ll = cost(pY, Ytest_ind)
                LL_nesterov.append(ll)
                print "Cost at iteration i = %d, j = %d : %.6f" %(i, j,ll)
                err = error_rate(pY, Ytest)
                CR_nesterov.append(err)
                print "Error rate:", err
                
    pY, _ = forward(Xtest, W1, b1, W2, b2)
    err = error_rate(pY, Ytest)
    
    