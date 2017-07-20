"""
* Module:
* Sequential:
* Linear:
* Tanh:
* ReLU:
* Sigmoid:
* Log
* LogSoftMax:
* SoftMax:
- gradient_check
* MSE:
* CrossEntropy:
* NLL:

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pickle as pkl
import nn

from sklearn.utils import shuffle
with open('mnist/data.pkl', 'rb') as f:
	data = pkl.load(f)
X_raw = (data['X'] - 128.) / 128.
y_raw = data['y']

from sklearn.model_selection import train_test_split as tts
X_train, X_test, y_train, y_test = tts(X_raw, y_raw, test_size = .2, random_state = 42)		#20% of data = leave for test
X_train, X_val, y_train, y_val = tts(X_train, y_train, test_size = .25, random_state = 42) 	#20% of data = leave for validation
#print (X_train.shape, y_train.shape)
#print (X_val.shape, y_val.shape)
#print (X_test.shape, y_test.shape)

from sklearn.metrics import accuracy_score

NEPOCH = 40
NITER = 10000
BSIZE = X_train.shape[0] // NITER


def main(X_train, y_train, X_test, y_test, X_val, y_val):
	model = nn.Sequential()
	model.add(nn.Linear(784, 10))
	#model.add(nn.Sigmoid(10, 10))
	model.add(nn.CrossEntropyCriterion())

	predictions_train = model.forward(inputs=X_train, target=y_train, return_loss=False)
	accuracy_train = accuracy_score((predictions_train == np.amax(predictions_train, axis=1).reshape((-1, 1))).astype(int), y_train)
   
	print ('Prediction accuracy at start: %-5.5f' % accuracy_train)

	learning_rate = 1e-3
	for epoch in xrange(NEPOCH):
		X, y = shuffle(X_train, y_train)
		avg_loss = 0
		
		if epoch == NEPOCH // 2:
			learning_rate /= 2
		elif epoch == 3 * NEPOCH // 4:
			learning_rate /= 2

		for it in xrange(NITER):
			inputs = X[it:it+BSIZE,:]
			target = y[it:it+BSIZE,:]
			loss = model.forward(inputs=inputs, target=target)
			avg_loss += loss
			#model.backward(1e-3 / np.sqrt(epoch + 1))
			model.backward(learning_rate)

		avg_loss /= NITER
		gcheck = model.gradient_check('Linear', inputs, target)[-1]
		
		predictions_train = model.forward(inputs=X_train, target=y_train, return_loss=False)
		accuracy_train = accuracy_score((predictions_train == np.amax(predictions_train, axis=1).reshape((-1, 1))).astype(int), y_train)
   
		predictions_val = model.forward(inputs=X_val, target=y_val, return_loss=False)
		accuracy_val = accuracy_score((predictions_val == np.amax(predictions_val, axis=1).reshape((-1, 1))).astype(int), y_val)
		
		print ('Epoch: %-5dLoss: %-10.5fGrads: %-10.5fAccuracy (train): %-10.5fAccuracy (val): %-10.5f' % (epoch, avg_loss, gcheck, accuracy_train, accuracy_val))

	predictions_test = model.forward(inputs=X_test, target=y_test, return_loss=False)
	accuracy_test = accuracy_score((predictions_test == np.amax(predictions_test, axis=1).reshape((-1, 1))).astype(int), y_test)
	print ('Final accuracy on test_set: %-10.5f' % accuracy_test)  

	return True  	

if __name__ == '__main__':
	main(X_train, y_train, X_test, y_test, X_val, y_val)
