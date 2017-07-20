from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nn.module import Module
import numpy as np

class NLL(Module):
    def __init__(self, *args):
        super(NLL, self).__init__(*args)
        self.name = "NLL"
        self.need_target = True
        self.param = None

    def log(self, z):
        return np.log(1e-5 + z)

    def forward(self, *args, **kwargs):
        #print ('f-prop called for module %s' % self.name)
        super(NLL, self).forward(*args, **kwargs)
        self.answers = self.input
        self.output = - np.multiply(self.target, self.log(self.input)).sum().sum()
        return self.output
    
    def backward(self, *args, **kwargs):
        #print ('b-prop called for module %s' % self.name)
    	super(NLL, self).backward(*args, **kwargs)
        #print ('b-prop called for module %s; got next_grad with shape %s; grad wrt to input shape: %s' % (self.name, self.next_grad.shape, self.grad_input.shape))
        return self.grad_input
    	
    def update_grad_input(self, *args, **kwargs):
    	self.grad_input = - self.target / self.input
        return self.grad_input

    def update_parameters(self, next_grad, learning_rate):
        self.next_grad = next_grad
        pass

    def gradient_check_local(self, *args):
        pass

    def local_gradient(self, inputs, target=None, eps=1e-3, tol=1e-3):
        self.forward(inputs, target)
        #===========================
        grad_an = - target / inputs
        #===========================
        grad_num = np.zeros(shape=np.prod(inputs.shape))
        
        for j, x in enumerate(np.nditer(inputs, op_flags = ['readwrite'])):
            x -= eps
            left = self.forward(inputs, target)
            x += 2 * eps
            right = self.forward(inputs, target)
            x -= eps # set to initial values
            der = (right - left) / (2. * eps)
            grad_num[j] = der
        grad_num = np.reshape(np.array(grad_num), newshape=grad_an.shape)
        #print (grad_num,'\n', grad_an)
        norm = np.linalg.norm(grad_an - grad_num) / np.prod(grad_an.shape)
        print ("||Grad_input_num - Grad_input_an|| / Size = %.6f" % (norm))
        
        if norm < tol:
            return True
        else:
            return False