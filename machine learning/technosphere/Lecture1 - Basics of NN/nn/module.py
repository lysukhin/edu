from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

class Module(object):
    def __init__(self, input_size=None, output_size=None):
        self.input_size = input_size
        self.output_size = output_size
        self.output = None
        self.grad_input = None

    def forward(self, inputs, target=None, batch_size=None, lambda_reg=0):
        if type(inputs) == list:
            self.input = np.array(input)
        else:
            self.input = inputs
        self.batch_size = batch_size
        self.lambda_reg = lambda_reg

        if target is not None:
            if type(target) == list:
                self.target = np.array(target)
            else:
                self.target = target

    def backward(self, next_grad, learning_rate):
        self.update_grad_input(next_grad, learning_rate)
        self.update_parameters(next_grad, learning_rate)

    def update_grad_input(self, *args, **kwargs):
        pass
        #raise NotImplementedError('implement computation of gradient w.r.t. input! df(x)/dx!')

    def update_parameters(self, next_gradient):
        # that's fine not to implement this method
        # module may have not parameters (for example - MSE criterion)
        pass

    def gradient_check(self, *args):
        pass