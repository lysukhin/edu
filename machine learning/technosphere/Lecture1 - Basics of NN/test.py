import numpy as np

def softmax(z):
	print z
	K = np.amax(z, axis=1).reshape((-1, 1))
	print K
	exp_z = np.exp(z - K)
	print exp_z
	return exp_z / exp_z.sum(axis=1).reshape((-1, 1))

#X = np.array([[1., 2, 3],
#	 [2, 3., 4.]])

#print softmax(X)

#print (X == np.amax(X, axis=1).reshape((-1, 1))).astype(int)

def relu(z):
	return np.multiply(z, (z > 0).astype(int))

#print relu(X)

import nn

X = np.random.normal(10, 1., size = (100, 100))
y = np.random.normal(.5, .25, size = (100, 100))

logg = nn.NLL()
logg.local_gradient(X, y)