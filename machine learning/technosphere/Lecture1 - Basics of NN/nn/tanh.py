from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nn.module import Module
import numpy as np
import math

class Tanh(Module):
	def __init__(self, *args):
		super(Tanh, self).__init__(*args)
		self.name = "Tanh"
		self.param = None
		self.need_target = False
		assert self.input_size == self.output_size

	def sigm(self, z):
		return 1. / (1. + np.exp(-z))

	def tanh(self, z):
		return 2 * self.sigm(2 * z) - 1
		
	def forward(self, *args):
		super(Tanh, self).forward(*args)
		self.output = self.tanh(self.input)
		#print ('f-prop called for module %s' % self.name)
		return self.output
	
	def backward(self, *args, **kwargs):
		super(Tanh, self).backward(*args, **kwargs)
		#print ('b-prop called for module %s; got next_grad with shape %s; grad wrt to input shape: %s' % (self.name, self.next_grad.shape, self.grad_input.shape))
		return np.multiply(self.next_grad, self.grad_input)
		
	def update_grad_input(self, *args, **kwargs):
		self.grad_input = 4 * np.multiply(self.sigm(2 * self.input), 1 - self.sigm(2 * self.input))
		
	def update_parameters(self, next_grad, learning_rate):
		self.next_grad = next_grad
		pass

	def local_gradient(self, inputs, target=None, eps=1e-3, tol=1e-3):
		self.forward(inputs, target)
		#===========================
		grad_an = 4 * np.multiply(self.sigm(2 * inputs), 1 - self.sigm(2 * inputs))
		#===========================
		grad_num = np.zeros(shape=np.prod(inputs.shape))
		
		for j, x in enumerate(np.nditer(inputs, op_flags = ['readwrite'])):
			x -= eps
			left = self.forward(inputs, target)
			x += 2 * eps
			right = self.forward(inputs, target)
			x -= eps # set to initial values
			der = (right - left) / (2. * eps)
			grad_num[j] = der.ravel()[j]
		grad_num = np.reshape(np.array(grad_num), newshape=grad_an.shape)
		#print (grad_num,'\n', grad_an)
		norm = np.linalg.norm(grad_an - grad_num) / np.prod(grad_an.shape)
		print ("||Grad_input_num - Grad_input_an|| / Size = %.6f" % (norm))
		
		if norm < tol:
			return True
		else:
			return False