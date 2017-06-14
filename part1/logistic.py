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
        Tvalid = y2indicator(Yvalid)
        N,D = X.shape
        K = len(set(Y))
        T = y2indicator(Y)
        
        self.W = np.random.randn(D, K)/ np.sqrt(D+K)
        self.b = np.zeros(K)
        
        costs = []
        best_valiidation_error = 1
        
        for i in xrange(epochs):
            pY = self.forward(X)
            
            #gradient descent
            self.W -= learning_rate*(X.T.dot(pY-T) + reg*self.W)
            self.b -= learning_rate*((pY-T).sum(axis = 0) + reg*self.b)
            
            if i%10 == 0:
                pYvalid = self.forward(Xvalid)
                c = cost(Tvalid, pYvalid)
                costs.append(c)
                e = error_rate(Tvalid, np.argmax(pYvalid, axis = 1))
                print "i:", i, 
                
                if e < best_valiidation_error:
                    best_valiidation_error = e
                    
        print "best validation error:", best_valiidation_error
        
        if show_fig:
            plt.plot(costs)
            plt.show()
            
    def forward(self, X):
        return softmax(X.dot(self.W)+self.b)
        
    def predict(self, X):
        pY = self.forward(X)
        return np.argmax(pY, axis = 1)
        
    def score(self, X, Y):
        prediction = self.predict(X)
        return 1 - error_rate(Y, prediction)
        
def main():
    X, Y = getData()
    model = LogisticModel()
    model.fit(X, Y)
    print model.score(X, Y)

if __name__ == '__main__':
main()