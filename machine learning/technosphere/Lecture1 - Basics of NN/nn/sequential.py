from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nn.module import Module
import numpy as np


class Sequential(Module):
    def __init__(self, n_epochs=None, learning_rate=1e-3, tol=1e-3, lambda_reg=0):
        self.layers = []
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.tol = tol
        self.lambda_reg = lambda_reg

    def add(self, module, name=None):
        if name:
            module.name = name
        self.layers.append(module)  

    def remove(self, module_name):
        found = False
        for layer in self.layers:
            if layer.name == module_name:
                found = True
                self.layers.remove(layer)
                return True
        if not found:
            print ("nothing to remove, though")
            return False

    def forward(self, inputs, target, return_loss=True):
        batch_size = inputs.shape[0]
        for j, layer in enumerate(self.layers):
            inputs = layer.forward(inputs, target, batch_size, self.lambda_reg)
        self.loss = inputs
        if return_loss:
            return self.loss
        else:
            return self.layers[-1].answers

    def backward(self, learning_rate=None, lambda_reg=0):
        if learning_rate is None:
            learning_rate = self.learning_rate
        if lambda_reg != 0:
            self.lambda_reg = lambda_reg
        next_grad = np.array([0])
        for j, layer in enumerate(self.layers[::-1]):
            next_grad = layer.backward(next_grad, learning_rate)

    def gradient_check(self, module_name, inputs, target, eps = 1e-3, tol = 1e-3):
        
        if type(inputs) == list:
            self.inputs = np.array(inputs)
        else:
            self.inputs = inputs
        
        found = False

        for layer in self.layers:
            if layer.name == module_name:
                found = True
                gradient_analytical = layer.gradient_check_local()
                gradient_numerical = []
                for weight in np.nditer(layer.param, op_flags = ['readwrite']):
                    weight[...] -= eps
                    left = self.forward(inputs, target)
                    weight[...] += 2 * eps
                    right = self.forward(inputs, target)
                    weight[...] -= eps # set to initial values
                    gradient_numerical.append((right - left) / (2. * eps))
                gradient_numerical = np.reshape(np.array(gradient_numerical), newshape=gradient_analytical.shape)
                return [gradient_numerical, 
                        gradient_analytical,
                        np.linalg.norm(gradient_numerical - gradient_analytical) / gradient_analytical.size]
        if not found:
            return [-1, -1, -1]