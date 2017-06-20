# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 11:16:41 2017

@author: pj
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 11:37:57 2017

@author: pj
"""

import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from util import init_weight, get_robert_frost

class SimpleRNN:
    def __init__(self, D, M, V):
        self.D = D
        self.M = M
        self.V = V
        
    def fit(self, X, learning_rate=10e-1, mu=0.99, reg=1.0, activation=T.tanh, epochs=500, show_fig=False):
        N = len(X)
        D = self.D
        M = self.M
        V = self.V
        self.f = activation
        
        #initialize weights
        We = init_weight(V, D)
        Wx = init_weight(D, M)
        Wh = init_weight(M, M)
        bh = np.zeros(M)
        h0 = np.zeros(M)
        z = np.ones(M)
        Wo = init_weight(M, V)
        bo = np.zeros(V)
        
        
        thX, thY, py_x, prediction = set(We, Wx, Wh, bh, h0, z, Wo, bo, activation)
        cost = -T.mean(T.log(py_x[T.arange(thY.shape[0]), thY]))
        grads = T.grad(cost, self.params)
        dparams = [theano.shared(p.get_value()*0) for p in self.params]
        
        updates = [
            (p, p + mu*dp - learning_rate*g) for p, dp, g in zip(self.params, dparams, grads)   
        ] + [
            (dp, mu*dp - learning_rate*g) for dp, g in zip(dparams, grads)        
        ]
        
        self.predict_op = theano.function(inputs=[thX], outputs=prediction)
        self.train_op = theano.function(
            inputs=[thX,thY],
            outputs=[cost,prediction],
            updates=updates,
        )
        
        
        costs = []
        n_total = sum((len(sentence) + 1) for sentence in X)#one for start and end tokens
        for i in xrange(epochs):
            X = shuffle(X)
            n_correct = 0
            n_total = 0
            cost = 0
            for j in xrange(N):
                if np.random.random() < 0.1: #only 10% of the time, we go to end of the sequence
                    input_sequence = [0] + X[j]
                    output_sequence = X[j] + [1]
                else:
                    input_sequence = [0] + X[j][:-1]
                    output_sequence = X[j] 
                    
                n_total += len(output_sequence)
                c, p = self.train_op(input_sequence, output_sequence)
                cost += c
                for pj, xj in zip(p, output_sequence):
                    if pj==xj:
                        n_correct += 1
            print "i:", i, "cost:", cost, "correct rate:", (float(n_correct)/n_total)
            costs.append(cost)
        if show_fig:
            plt.plot(costs)
            plt.show()
            
    def save(self, filename):
        np.savez(filename, *[p.get_value() for p in self.params])
        #this is eq of passing each param separately
        #savez used to save multiple arrays
        
        
    @staticmethod
    def load(filename, activation):
        npz = np.load(filename)
        We = npz['arr_0']
        Wx = npz['arr_1']
        Wh = npz['arr_2']
        bh = npz['arr_3']
        h0 = npz['arr_4']
        z = npz['arr_5']
        Wo = npz['arr_6']
        bo = npz['arr_7']
        
        V, D = We.shape
        _, M = Wx.shape
        rnn = SimpleRNN(D, M, V)
        rnn.set(We, Wx, Wh, bh, h0, z, Wo, bo, activation)
        
        return rnn
        
    def set(self, We, Wx, Wh, bh, h0, z, Wo, bo, activation):
        self.f = activation
        self.We = theano.shared(We)
        self.Wx = theano.shared(Wx)
        self.Wh = theano.shared(Wh)
        self.bh = theano.shared(bh)
        self.Wo = theano.shared(Wo)
        self.bo = theano.shared(bo)
        self.h0 = theano.shared(h0)
        self.z = theano.shared(z)
        
        self.params = [self.We, self.Wx, self.Wh, self.bh, self.h0, self.z, self.Wo, self.bo]
        
        thX = T.ivector('X') #just sequences of indexes
        Ei = self.We[thX] #real X, T x D mat. D is size of word embedding, T is length of sequence
        
        thY = T.ivector('Y')
        
        def recurrence(x_t, h_t1):
            # returns h(t), y(t)
            hhat_t = self.f(x_t.dot(self.Wx) + h_t1.dot(self.Wh) + self.bh)
            h_t = (1-self.z)*(h_t1) + self.z*hhat_t
            y_t = T.nnet.softmax(h_t.dot(self.Wo) + self.bo)
            return h_t, y_t
            
        [h,y], _ = theano.scan(
            fn=recurrence,
            outputs_info=[self.h0,None],
            sequences=Ei,
            n_steps=Ei.shape[0],            
            )
        
        py_x = y[:,0,:]
        prediction = T.argmax(py_x, axis=1)
        self.predict_op = theano.function(
            inputs=[thX], 
            outputs=[py_x,prediction],
            allow_input_downcast=True,            
            )
        return thX, thY, py_x, prediction
            
    def generate(self, word2idx):
        idx2word = {v:k for k,v in word2idx.iteritems()}
        V = len(word2idx)
        
        n_lines = 0
        #four lines at a time
        
        #initial word
        X = [0]
        print idx2word[X[0]],
        
        while n_lines <4:
            PY_X, _ = self.predict_op(X)
            PY_X = PY_X[-1].flatten()
            P = [np.random.choice(V, p=PY_X)]
            X += P
            
            #p is an array
            #get just last predicted word
            
            
            if P > 1:
                word = idx2word[P]
                print word
                
            elif P==1:
                n_lines += 1
                X = [0]
                print ''
               
def train_poetry():
    sentences, word2idx = get_robert_frost()
    rnn = SimpleRNN(50,50, len(word2idx))
    rnn.fit(sentences, learning_rate=10e-5,show_fig=True,activation=T.nnet.relu, epochs=2000)
    rnn.save('RRNN_D50_M50_epcohs2000.npz')
    
def generate_poetry():
    sentences, word2idx = get_robert_frost()
    rnn = SimpleRNN.load('RRNN_D50_M50_epcohs2000.npz')   
    rnn.generate(pi, word2idx)
    
    
if __name__ == "__main__":
    train_poetry()
    generate_poetry()
    
    
    