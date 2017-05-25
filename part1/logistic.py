# -*- coding: utf-8 -*-
"""
Created on Thu May 25 16:38:38 2017

@author: pj

logistic on facial recognition
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from util import getData, softmax, cost, y2indicator, error_rate


class LogisticModel(object):
    def __init__(self):
        pass
    def fit(self, X, Y, learning_rate = 10e-8, reg = 10e-12, epochs = 10000, show_fig = False):
        X, Y = shuffle(X, Y)
        Xvalid, Yvalid = X[-1000:], Y[-1000:]
        X, Y = X[:-1000], Y[:-1000]
        
        N,D = X.shape
        K = len(set(Y))
        T = y2indicator(Y)
        
        self.W = np.random.randn(D, K)/ np.sqrt(D+K)
        self.b = np.zeros(K)
        
        cost = []
        best_valiidation_error = 1
        
        for