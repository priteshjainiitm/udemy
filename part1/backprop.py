# -*- coding: utf-8 -*-
"""
Created on Mon May 22 05:22:56 2017

@author: pj
"""

# -*- coding: utf-8 -*-
"""
Created on Sun May 21 14:34:07 2017

@author: pj
"""

import numpy as np
import matplotlib.pyplot as plt
import os

os.chdir('/home/pj/udemy/part1')

def main():
    Nclass = 500
    
    x1 = np.random.randn(Nclass, 2) + np.array([0, -2])
    x2 = np.random.randn(Nclass, 2) + np.array([2, 2])
    x3 = np.random.randn(Nclass, 2) + np.array([-2, 2])
    X = np.vstack([x1,x2,x3])
    
    Y = np.array([0]*Nclass + [1]*Nclass + [2]*Nclass)
    
    D = 2
    K = 3
    M = 3
    N = len(Y)
    T = np.zeros((N, K))
    
    for i in xrange(N):
        T[i, Y[i]] = 1
    
    plt.scatter(X[:,0], X[:,1], c = Y, s= 100, alpha = 0.5)

    W1 = np.random.randn(D, M)
    b1 = np.random.randn(M)
    W2 = np.random.randn(M, K)
    b2 = np.random.randn(K)
    
    learning_rate = 10e-7

    costs = []
    
    for epoch in xrange(100000):
        output, hidden = forward(X, W1, b1, W2, b2)
        if epoch % 100 == 0:
            c = cost(T, output)
            P = np.argmax(output, axis = 1)
            r = classification_rate(Y, P)
            print 'cost:', c, 'classificaiton_rate', r
            costs.append(c)
            
        W2 += learning_rate*derivative_W2(hidden, T, output)
        b2 += learning_rate*derivative_b2(T, output)
        W1 += learning_rate*derivative_W1(X, hidden, T, output, W2)
        b1 += learning_rate*derivative_b1(T, output, W2, hidden)
        
        #plt.plot(costs)
        #plt.show()
        
def derivative_W2(Z, T, Y):
    N, K = T.shape
    M = Z.shape[1]
    '''
    ret1 = np.zeros((M, K))
    
    
    #slow way
    for n in xrange(N):
        for m in xrange(M):
            for k in xrange(K):
                ret1[m,k] += (T[n,k] - Y[n,k])*Z[n,m]
                
    
                
    #fast
    #m appears only once both sides
    ret2 = np.zeros((M, K))
    for n in xrange(N):
        for k in xrange(K):
            ret2[:, k] += (T[n,k] - Y[n,k])*Z[n,:]
            
    
    
    ret3 = np.zeros((M, K))
    
    for n in xrange(N):
        ret3 += np.outer(Z[n], T[n]-Y[n])
    
    ret4 = Z.T.dot(T-Y)
            
    #assert(abs(ret3 - ret4).sum() < 10e-5)
    '''
    return Z.T.dot(T-Y)
    
def derivative_b2(T, Y):
    return (T-Y).sum(axis = 0)
    
def derivative_W1(X, Z, T, Y, W2):
    N, D = X.shape
    M, K = W2.shape
    
    '''
    #slow
    ret1 = np.zeros((D, M))
    
    for n in xrange(N):
        for k in xrange(K):
            for m in xrange(M):
                for d in xrange(D):
                    ret1 += (T[n,k] - Y[n,k])*W2[m,k]*Z[n,m]*(1-Z[n,m])*X[n,d]
    return ret1
    '''
    dz = (T - Y).dot(W2.T)*Z*(1-Z)
    return X.T.dot(dz)
    
def derivative_b1(T, Y, W2, Z):
    return ((T-Y).dot(W2.T)*Z*(1-Z)).sum(axis = 0)
                    
    
def cost(T, Y):
    tot = T*np.log(Y)
    return tot.sum()

def forward(X, W1, b1, W2, b2):
    #sigmod
    z = 1/ (1 + np.exp(-X.dot(W1) - b1))
    A = z.dot(W2) + b2
    expA = np.exp(A)
    Y = expA/expA.sum(axis = 1, keepdims = True)
    return z, Y
    
    
def  classification_rate(Y, P):
    n_correct = 0
    n_total = 0
    for i in xrange(len(Y)):
        n_total += 1
        if Y[i] == P[i]:
            n_correct += 1           
    return float(n_correct)/n_total
    
#P_Y_given_X = forward(X, W1, b1, W2, b2)

#P = np.argmax(P_Y_given_X, axis = 1)

#print classification_rate(Y, P)



if __name__ == 'main':
    main()
#assert(len(P)==len(Y))
