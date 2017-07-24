# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 16:10:13 2017

@author: pj
"""

import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from util import relu, error_rate, getKaggleMNIST, init_weights


class Layer(object):
    def __init__(self, m1, m2):
        W = init_weights((m1,m2))
        bi = np.zeros(m1)
        bo = np.zeros(m2)        
        self.W = theano.shared(W)
        self.bi = theano.shared(bi)
        self.bo = theano.shared(bo)        
        self.params = [self.W, self.bi, self.bo]

    def forward(self, X):
        return T.nnect.sigmoid(X.dot(self.W) + self.bi)

    def forwardT(self, X):
        return T.nnet.sigmoid(X.dot(self.W.T) + self.bo)

class DeepAutoEncoder(object):
    def __init__(self, hidden_layer_sizes):
        self.hidden_layer_sizes = hidden_layer_sizes
        
    def fit(self, X, learning_rate=0.5, mu=0.99, epochs=50, batch_sz=100, show_fig=False):
        N, D = X.shape
        n_batches = N / batch_sz
        
        mi = D
        self.layers = []
        self.params = []
        for mo in self.hidden_layer_sizes:
            layer = Layer(mi, mo)
            self.layers.append(layer)
            self.params += layer.params
            mi = mo
            
        X_in = T.matrix('X')
        X_hat = self.forward(X_in)
        
        cost = -(X_in*T.log(X_hat) + (1-X_in)*T.log(1-X_hat)).mean()
        cost_op = theano.function(
            inputs=[X_in],
            outputs = cost,        
        )
        
        dparams = [theano.shared(p.get_value()*0) for p in self.params]

        grads = T.grads(cost, self.params)        
        
        updates = [
            (p, p+mu*dp - learning_rate*g) for p, dp, g in zip(params, dparams, grads)        
        ] + [
            (dp, mu*dp - learning_rate*g) for dp, g in zip(dparams, grads)        
        ]
        
        train_op = theano.function(
            inputs=[X_in],
            outputs=cost,
            updates=updates,
        )
        
        costs = []
        
        for i in xrange(epochs):
            print "epoch:", i
            X = shuffle(X)
            for j in xrange(n_batches):
                batch = X[J*batch_sz:(j*batch_sz + batch_sz)]
                c = train_op(batch)
                if j%100 == 0:
                    print "j / n_batches:", j, "/", n_batches, "cost:", c
                costs.append(c)
                
        if show_fig:
            plt.plot(costs)
            plt.show()
            
    def forward(self, X):
        Z = X
        for layer in self.layers:
            Z = layer.forward(Z)
            
        self.map2center = theano.function(
            inputs = [X],
            outputs = Z,
        )
        
        for i in xrange(len(self.layers) -1, -1, -1):
            Z = self.layers[i].forwardT(Z)
        return Z
        
def main():
    Xtrain, Ytrain, Xtest, Ytest = getKaggleMNIST()
    dae = DeepAutoEncoder([500,300,2])
    dae.fit(Xtrain)
    mapping = dae.map2center(Xtrain)
    plt.scatter(mapping[:,0], mapping[:,1], c=Ytrain, s=100,alpha=0.5)
    plt.show()

if __name__ == '__main__':
    main()