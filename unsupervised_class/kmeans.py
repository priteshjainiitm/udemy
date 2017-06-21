# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 15:38:54 2017

@author: pj
"""

import numpy as np
import matplotlib.pyplot as plt

def d(u,v):
    diff = u-v
    return diff.dot(diff)
    
def cost(X, R, M):
    cost = 0
    for k in xrange(M):
        for n in xrange(N):
            cost += R[n,K]*d(M[k], X[n])
            


def plot_k_means(X, K, max_iter = 30, beta = 1.0):
    N, D = X.shape
    M = np.zeros((K, D))
    R = np.zeros((N,K))
    
    for k in xrange(K):
        M[k] = X[np.random.choice(N)]
    costs = np.zeros(max_iter)
    for i in xrange(max_iter):
        for k in xrange(K):
            for n in xrange(N):
                R[n,K] = np.exp(-beta*d(M[k], X[n]))/np.sum(np.exp(-beta*d(M[j], X[n] )) for j in xrange(K))
      
    
        for k in xrange(k):
            M[k] = R[:, k].dot(X) / R[:, k].sum()
            
        costs[i] = cost[X, R, M]
        if i > 0:
            if np.abs(costs[i] - costs[i-1]) < 0.1:
                break
    plt.plot(costs)
    plt.title("Costs")
    plt.show()
    
    random_colors = np.random.random((K,3))
    colors = R.dot(random_colors)
    plt.scatter(X[:,0], X[:,1], c=colors)
    plt.show()
        
    
def main():
    D = 2
    s = 4
    mu1 = np.array([0,0])
    mu2 = np.array([s,s])
    mu3 = np.array([0,s])

    N = 900
    X = np.zeros((N,D))
    X[:300, :] = np.random.randn(300,D) + mu1    
    X[300:600, :] = np.random.randn(300,D) + mu2   
    X[600:, :] = np.random.randn(300,D) + mu3
    
    plt.scatter(X[:,0], X[:,1])
    plt.show()
    
    K = 3
    plot_k_means(X, K)
    
    K = 5
    plot_k_means(X,K, max_iter = 30)
    
    K = 5
    plot_k_means(X, K, max_iter = 30, beta = 0.3)
    

    
    
    
if __name__ =="__main__":
    main()