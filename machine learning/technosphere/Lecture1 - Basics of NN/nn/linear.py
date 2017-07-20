from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nn.module import Module
import numpy as np

class Linear(Module):
    def __init__(self, *args):
        """
        N_features -> K_classes
        """
        super(Linear, self).__init__(*args)
        self.name = "Linear"
        self.weights = np.random.normal(loc = 0., scale = 1. / np.sqrt(self.input_size), size = (self.output_size, self.input_size))	
        self.bias = np.zeros(shape = (self.output_size, 1))
        self.param = self.weights
        self.need_target = False

    def forward(self, *args):
        super(Linear, self).forward(*args)
        self.output = self.input.dot(self.weights.T) + np.ones(shape = (self.batch_size, 1)).dot(self.bias.T)
        #print ('f-prop called for module %s' % self.name)
        return self.output
    
    def backward(self, *args, **kwargs):
        self.grad_output_weights = self.input
        self.grad_output_bias = np.ones(shape = (self.batch_size, 1))
        super(Linear, self).backward(*args, **kwargs)
    	#print ('b-prop called for module %s; got next_grad with shape %s; grad wrt to input shape: %s' % (self.name, self.next_grad.shape, self.grad_input.shape))
        return self.next_grad.dot(self.grad_input)

    def update_grad_input(self, *args, **kwargs):
    	self.grad_input = self.weights

    def update_parameters(self, next_grad, learning_rate):
        self.next_grad = next_grad
        self.weights -= learning_rate * (self.next_grad.T.dot(self.grad_output_weights) + self.lambda_reg * self.weights)
        self.bias -= learning_rate * self.next_grad.T.dot(self.grad_output_bias)

    def gradient_check_local(self, *args):
        return self.next_grad.T.dot(self.grad_output_weights)