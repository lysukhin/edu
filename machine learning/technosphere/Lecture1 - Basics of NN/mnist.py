from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original', data_home="./mnist")
print (mnist.data.shape)
print (mnist.target.shape)

import numpy as np
y = np.zeros(shape=(mnist.data.shape[0],
					np.unique(mnist.target).shape[0]))
print (y.shape)

print mnist.target

for s,t in zip(y, mnist.target):
	s[int(t)] = 1

print y

data = {'X' : mnist.data,
		'y' : y }

import pickle as pkl
with open("mnist/data.pkl", 'wb') as f:
	pkl.dump(data, f, pkl.HIGHEST_PROTOCOL)
