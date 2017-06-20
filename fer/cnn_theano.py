#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 20:06:27 2017

@author: pj
"""

import numpy as np
import theano
import theano.tensor as T
from sklearn.utils import shuffle
from theano.tensor.nnet import conv2d
from theano.tensor.signal import downsample
import matplotlib.pyplot as plt

from util import getImageDate, error_rate, init_weight_and_bias, init_filter
from ann_theano import HiddenLayer

class ConvPoolLayer(object):
    def __init__(self, mi, mo,fw=5, fh=5, poolsz=(2,2)):
        sz = (mo, mi, fw, fh)#theano filter order requirement
        W0 = init_filter(sz, poolsz)
        self.W = theano.shared(W0)
        b0 = np.zeros(mo, dtype=np.float32)
        self.b = theano.shared(b0)
        self.poolsz = poolsz
        self.params =   [self.W, self.b]
        
    def forward(self, X):
        conv_out = conv2d(input=X, filters = self.W)
        pooled_out = downsample.max_pool_2d(
            input = conv_out,
            ds = self.poolsz,
            ignore_border=True,
                )
        return T.tanh(pooled_out + self.dimshuffle('x', 0, 'x','x'))

class CNN(object):
    def __init__(self, convpool_layer_sizes, hidden_layer_sizes):
        self.convpool_layer_sizes = convpool_layer_sizes
        self.hidden_layer_sizes = hidden_layer_sizes
        
    def fit(self, X, Y, lr=10e-5, mu=0.99, reg=10e-7, decay=0.99999, eps=10e-3, batch_sz=100, epochs = 200, show_fig=False):
        lr = np.float32(lr)
        mu = np.float32(mu)
        reg = np.float32(reg)
        eps = np.float32(eps)
        decay = np.float32(decay)
        
        X, Y = shuffle(X,Y)
        X = X.astype(np.float32)
        Y = Y.astype(np.int32)
        
        Xvalid, Yvalid = X[-1000:], Y[-1000:]
        X,Y = X[:-1000], Y[:-1000]
        
        N, c, d, d = X.shape
        mi = c
        outw = d
        outh = d
        self.convpool_layers = []
        for mo, fw, fh in self.convpool_layer_sizes:
            layer = ConvPoolLayer(mi, mo, fw, fh)
            self.convpool_layers.append(layer)
            outw = (outw - fw + 1)/2
            outh = (outh - fh + 1)/2
            mi = mo
            
        K = len(set(Y))
        self.hidden_layers = []
        M1 = self.convpool_layer_sizes[-1][0]*outw*outh
        count = 0
        for M2 in self.hidden_layer_sizes:
            h = HiddenLayer(M1, M2, count)
            self.hidden_layers.append(h)
            M1 = M2
            count += 1
        W, b = init_weight_and_bias(M1, K)
        self.W = theano.shared(W, 'W_logreg')
        self.b = theano.shared(b, 'b_logreg')
        
        
        self.params = [self.W, self.b]
        for c in self.convpool_layers:
            self.params += c.params
            
        for h in self.hidden_layers:
            self.params += h.params
            
        # momentum
        dparams = [theano.shared(np.zeros(p.get_value().shape, dtype=np.float32)) for p in self.params]
        
        # for rmsprop 
        cache = [theano.shared(np.zeros(p.get_value().shape, dtype=np.float32)) for p in self.params]

        thX = T.tensor4('X',dtype = np.float32)
        thY = T.ivector('Y', dtype =  np.float32)
        pY = self.forward(thX)
        
        rcost = reg*T.sum([(p*p).sum() for p in self.params])
        cost = -T.mean(T.log(pY[T.arange(thY.shape[0]), thY])) + cost
        prediction = self.predict(thX)
        
        cost_predict_op = theano.function(inputs=[thX,thY], outputs=[cost, prediction])
        
        #doing only momentum
        updates = [
            (p, p + mu*dp - lr.T.grad(cost, p)) for p, dp in zip(self.params, dparams)
        ] + [
            (dp, mu*dp - lr.T.grad(cost, p)) for p, dp in zip(self.params, dparams)
                ]
                      
        train_op = theano.function(
                inputs = [thX, thY],
                updates = updates
                )
        
        
        n_batches = N/batch_sz
        costs = []
        for i in xrange(epochs):
            X, Y = shuffle(X,Y)
            for j in xrange(n_batches):
                Xbatch = X[j*batch_sz:(j*batch_sz+batch_sz)]
                Ybatch = Y[j*batch_sz:(j*batch_sz+batch_sz)]
                
                train_op(Xbatch, Ybatch)
                
                if j%20 == 0:
                    c, p = cost_predict_op(Xvalid, Yvalid)
                    costs.append(c)
                    e = error_rate(Yvalid, p)
                    print "i:", i, "j:", j, "nb:", n_batches, "cost:", c,   "error_rate:", e
        if show_fig:
            plt.plot(costs)
            plt.show()
            
    
    def forward(self, X):
        Z = X
        for c in self.convpool_layers:
            Z = c.forward(Z)
        Z = Z.flatten(ndim=2) #N by D after this
        
        for h in self.hidden_layers:
            Z = h.forward(Z)
        return T.nnet.softmax(Z.dot(self.W) + self.b)
    
    def predict(self, X):
        pY = self.forward(X)
        return T.argmax(pY, axis =1)
    
    

def main():
    X,Y = getImageData()
    
    model = CNN(
            convpool_layer_sizes=[(20,5,5), (20,5,5)],#feature_mapsxwidthxheight
            hidden_layer_sizes=[500,300],
            )
    model.fit(X,Y, show_fig==True)
    
if __name__ == '__main__':
    main()