#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 29 21:51:51 2017

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
    
    #1 const

    W1 = np.random.randn(D, M)/28
    b1 = np.zeros(M)
    W2 = np.random.randn(M, K)/ np.sqrt(M)
    b2 = np.zeros(K)
    
    LL_batch = []
    CR_batch = []
    
    for i in xrange(max_iter):
        for j in xrange(n_batches):
            Xbatch = Xtrain[j*batch_sz:(j*batch_sz + batch_sz),]
            Ybatch  = Ytrain_ind[j*batch_sz:(j*batch_sz+batch_sz)]
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
    
    #RMSprop
    W1 = np.random.randn(D, M)/28
    b1 = np.zeros(M)
    W2 = np.random.randn(M, K)/ np.sqrt(M)
    b2 = np.zeros(K)
    
    cache_W1 = 0
    cache_b1 = 0
    cache_W2 = 0
    cache_b2 = 0
    lr0 = 0.001
    decay_rate = 0.999
    eps = 0.000001
    LL_rms = []
    CR_rms = []
    
    for i in xrange(max_iter):
        for j in xrange(n_batches):
            Xbatch = Xtrain[j*batch_sz:(j*batch_sz + batch_sz),]
            Ybatch  = Ytrain[j*batch_sz:(j*batch_sz+batch_sz)]
            pYbatch, Z = forward(Xbatch, W1, b1, W2, b2)
            
            #updates
            gW2 = derivative_w2(Z, Ybatch, pYbatch) + reg*W2
            cache_W2 = decay_rate*cache_W2 + (1-decay_rate)*gW2*gW2
            W2 -= lr0*gW2/(np.qrt(cache_W2) + eps)                                 
                               
            gb2 = derivative_b2(Ybatch, pYbatch) + reg*b2
            cache_b2 = decay_rate*cache_b2 + (1-decay_rate)*gb2*gb2
            b2 -= lr0*gb2/(np.qrt(cache_b2) + eps)                                 
          
            gW1 = derivative_w1(Xbatch, Z, Ybatch, pYbatch, W2) + reg*W1
            cache_W1 = decay_rate*cache_W1 + (1-decay_rate)*gW1*gW1
            W2 -= lr0*gW1/(np.qrt(cache_W1) + eps)                                 
          
            gb1 = derivative_b1(Z, Ybatch, pYbatch, W2) + reg*b1
            cache_b1 = decay_rate*cache_b1 + (1-decay_rate)*gb1*gb1
            b2 -= lr0*gb1/(np.qrt(cache_b1) + eps)                                 
          
            
            if j% print_period ==0:
                pY, _ = forward(Xtest, W1, b1, W2, b2)
                ll = cost(pY, Ytest_ind)
                LL_rms.append(ll)
                print "Cost at iteration i = %d, j = %d : %.6f" %(i, j,ll)
                err = error_rate(pY, Ytest)
                CR_rms.append(err)
                print "Error rate:", err
                
    pY, _ = forward(Xtest, W1, b1, W2, b2)
    err = error_rate(pY, Ytest)
    
    plt.plot(LL_batch, label = 'Const')
    plt.plot(LL_rms, label = 'RMSprop')
    plt.legend()
    plt.show()
    
if __name__ == "__main__":
    main()