import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt

learning_rate = 0.01
num_iter = 1000

train_X = np.array([[100,105,110,117,122,121,125,134,140,123,143,147,148,155,156,152,156,183,198,201,196,194,146,161],
                    [100,107,114,122,131,138,149,163,176,185,198,208,216,226,236,244,266,298,335,336,387,407,417,431]])
train_Y = np.array([[100,101,112,122,124,122,143,152,151,126,155,159,153,177,184,169,189,225,227,223,218,231,179,240]])
n = train_X.shape[0]
m = train_X.shape[1]

X = tf.placeholder(tf.float32, [n, m])
Y = tf.placeholder(tf.float32, [1, m])

W = tf.Variable(tf.zeros([n ,1]))
alpha = W[0, :]
beta = W[1, :]
L = X[0, :]
K = X[1, :]

pred = beta * L**alpha * K**(1-alpha) 
cost = tf.reduce_sum(tf.square(pred-Y)) / (2*m)

train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)

for iteration in range(num_iter):
	session.run(train, feed_dict = {X: train_X, Y: train_Y})

# =============================================================================
print(session.run(W))
theta1 = 0.4
theta2 = 2
plt.plot([theta2*train_X[0]**theta1 * train_X[1]**(1-theta1)], train_Y, 'ro')
plt.xlabel("data")
plt.ylabel("price")
plt.show()
#print(train_X[0]+train_X[1])
# =============================================================================
# =============================================================================
# print(session.run(W))
# predict1 = tf.matmul(tf.transpose([[120.], [130.]]), W)
# print(session.run(predict1))
# =============================================================================


       		 