# -*- coding: utf-8 -*-
"""
Created on Fri May 26 12:09:07 2017

@author: pj
"""

import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from datetime import datetime

from util import get_transformed_data,
def main():
    X, Y, _, _ = get_transformed_data() #pca tranformed
    X = X[:, :300]
    
    #normoalize data
    mu = X.mean(axis = 0)
    std = X.std(axis = 0)
    X = (X-mu)/std
    
    #logistic regression
    Xtrain = X[:-1000,]
    Ytrain = Y[:-1000]
    Xtest = X[-1000:,]
    Ytest = Y[-1000:]
    
    N, D = X.shape
    Ytrain_ind = y2indicator(Ytrain)
    Ytest_ind = y2indicator(Ytest)
    
    # 1) full gradient descent
    W = np.random.randn(D, 10) /28
    b = np.zeros(10)
    
    LL = []
    lr = 0.0001
    reg = 0.01
    t0 = datetime.now()
    
    for i in xrange(200):
        p_y = forward(Xtrain, W,b)
        
        W += lr*(gradW(Ytrain_ind, p_y, Xtrain ) - reg*W)
        b += lr*(gradb(Ytrain_ind, p_y) - reg*b)
        
        p_y_test = forward(Xtest, W, b)
        ll = cost(p_y_test, Ytest_ind)
        LL.append(ll)
        
        if i%10 == 0:
            err = error_rate(p_y_test, Ytest)
            print "cost at iteration %d: %.6f" %(i, ll)
            print "error rate:", err
    print "final error rate:", error_rate(p_y_test, Ytest)
    print "final time elapsed for full GD:", datetime.now() - t0
    
    # 2) stochastic
    W = np.random.randn(D, 10)/28
    b = np.zeros(10)
    
    LL_stochastic = []
    lr = 0.0001
    reg = 0.01
    t0 = datetime.now()
    
    for i in xrange(1):
        tmpX, tmpY = shuffle(Xtrain, Ytrain_ind)
        for n in xrange(min(N, 500)):
            x = tmpX[n,:].reshape(1,D)
            y = tmpY[n,:].reshape(1, 10)
            p_y = forward(x, W, b)
            
            W += lr*(gradW(y, p_y, b) - reg*W)
            b += lr*(gradb(y, p_y) - reg*b)
            
            p_y_test = forward(Xtest, W, b)
            ll = cost(p_y_test, Ytest_ind)
            
            LL_stochastic.append
            
            if n%(N/2) == 0:
                err = error_rate(p_y_test, Ytest)
                print "cost at iteration %d: %.6f" %(i, ll)
                print "error rate:", err
    print "final error rate:", error_rate(p_y_test, Ytest)
    print "final time elapsed for full GD:", datetime.now() - t0
                
    # 3) batch
                
    
            
            