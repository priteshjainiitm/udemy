# -*- coding: utf-8 -*-
"""
Created on Sun May 21 15:34:03 2017

@author: pj
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def get_data():
    df = pd.read_csv('ecommerce_data.csv')
    data = df.as_matrix()
    
    X = data[:, :-1]
    Y = data[:, -1]
    
    #normalize the numerical columns
    X[:, 1] = (X[:, 1] - X[:, 1].mean())/X[:, 1].std()
    X[:, 1] = (X[:, 1] - X[:, 1].mean())/X[:, 1].std()

    N, D = X.shape
    X2 = np.zeros((N, D+3))
    X2[:, 0:(D-1)] = X[:, 0:D-1]
    
    for n in xrange(N):
        t = int(X[n, D-1])
        X2[n, t+t-1] = 1
        
    #another way
    Z = np.zeros((N,4))
    Z[np.arange(N), X[:,(D-1)].astype(np.int32)]  = 1
    #X2[:,-4:] = Z
    
    return X2, Y
    
#for logistic, we need only binary data
def get_binary_data():
    X,Y = get_data
    
    X2 = X[Y <= 1]
    Y2 = Y[Y <= 1]
    return X2, Y2
    
    