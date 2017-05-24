import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

Nclass = 500

x1 = np.random.randn(Nclass, 2) + np.array([0, -2])
x2 = np.random.randn(Nclass, 2) + np.array([2, 2])
x3 = np.random.randn(Nclass, 2) + np.array([-2, 2])
X = np.vstack([x1,x2,x3])

Y = np.array([0]*Nclass + [1]*Nclass + [2]*Nclass)

#plt.scatter(X[:,0], X[:,1], c = Y, s= 100, alpha = 0.5)

D = 2
K = 3
M = 3
N = len(Y)
T = np.zeros((N,K))
for i in xrange(N):
    T[i, Y[i]] = 1
     
     
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev = 0.01))

def forward(X,W1,b1,W2, b2):
    Z = tf.nn.sigmoid(tf.matmul(X,W1) + b1)
    #return only activations, not softmax
    return tf.matmul(Z,W2) + b2

tfX = tf.placeholder(tf.float32, [None, D])
tfY = tf.placeholder(tf.float32, [None, K])

W1 = init_weights([D, M])
b1 = init_weights([M])
W2 = init_weights([M, K])
b2 = init_weights([K])

py_x = forward(tfX, W1, b1, W2, b2)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = py_x,labels = tfY))

train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost)

predict_op = tf.argmax(py_x, 1)

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

for i in xrange(10000):
    sess.run(train_op, feed_dict={tfX: X, tfY: T})
    pred = sess.run(predict_op,feed_dict={tfX: X, tfY: T})
    if i%10 == 0:
        print np.mean(Y == pred)

