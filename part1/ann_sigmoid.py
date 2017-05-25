#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 09:05:34 2017

@author: pj
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from util import getBinaryData, sigmoid_cost, sigmoid, error_rate, relu

class ANN(object):
    def __init__(self, M):
        self.M = M
    
    def fit(self, X, Y, learning_rate = 5*10e-7, reg = 1.0, epochs = 10000, show_fig = False):
        
        X, Y = shuffle(X,Y)
        
        Xvalid, Yvalid = X[-1000:], Y[-1000:]
        X, Y = X[:-1000], Y[:-1000]
        
        N,D = X.shape
        
        self.W1 = np.random.randn(D, self.M) /np.sqrt(D + self.M)
        self.b1 = np.zeros(self.M)
        self.W2 = np.random.randn(self.M) / np.sqrt(self.M)
        self.b2 = 0
        
        costs = []
        
        best_validation_error = 1.0
        for i in xrange(epochs):
            pY, Z = self.forward(X)
            
            #gd
            pY_Y = pY - Y
            self.W2 -= learning_rate*(Z.T.dot(pY_Y) + reg*self.W2)
            self.b2 -= learning_rate*((pY_Y).sum() + reg*self.b2)
            
            dz = np.outer(pY_Y, self.W2)*(Z > 0)#relu der
            #dz = np.outer(pY_Y, self.W2)*(1 - Z*Z)#tanh der

            self.W1 -= learning_rate*(X.T.dot(dz) + reg*self.W1)
            self.b1 -= learning_rate*(np.sum(dz, axis = 0) + reg*self.b1)
            
            if i%20 == 0:
                pYvalid , _ = self.forward(Xvalid)
                c = sigmoid_cost(Yvalid, pYvalid)
                costs.append(c)
                e = error_rate(Yvalid, np.round(pYvalid))
                if e < best_validation_error:
                    best_validation_error = e
        print "best validation error:", best_validation_error
        if show_fig:
            plt.plot(costs)
            plt.show()
            
    def forward(self, X):
        Z = relu(X.dot(self.W1) + self.b1)
        #Z = np.tanh(X.dot(self.W1) + self.b1)

        return sigmoid(Z.dot(self.W2)+ self.b2), Z
    
    def predict(self, X):
        pY, _ = self.forward(self, X)
        return np.round(pY)
    
    def score(self, X, Y):
        prediction = self.predict(self, X)
        return 1 - error_rate(Y, prediction)
    
def main():
    X, Y = getBinaryData()
    
    X0, Y0 = X[Y!=1, :], Y[Y!=1]
    X1 = X[Y==1,:]
    X1 = np.repeat(X1, 9, axis = 0)
    X = np.vstack(X0, X1)
    Y = np.concatenate((Y0, [1]*len(X1)))
    
    model = ANN(100)
    model.fit(X,Y, show_fig = True)
    
if __name__ == '__main__':
    main()