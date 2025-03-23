from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api



def logsoftmax_chat(X, axis=-1):
    X_max = array_api.max(X, axis=axis, keepdims=True)  # For numerical stability
    exp_X = array_api.exp(X - X_max)  # Subtract max to prevent large exponentials
    sum_exp_X = array_api.sum(exp_X, axis=axis, keepdims=True)
    ## LogSoftMax == X - LogSumExp
    return X - X_max - array_api.log(sum_exp_X)


class LogSoftmax(TensorOp):

    def __init__(self):
      self.Z = None
      return

    def compute(self, Z):
      self.Z = Z
      lsm = logsoftmax_chat( Z, axis=1)
      return lsm      

    def gradient(self, out_grad, node):
      """Compute the gradient of LogSoftmax in terms of LogSoftmax itself."""
      lsm_z = node  ## log(softmax(z))
      sm_z = exp( lsm_z) ## softmax(z)  ## ops_mathematics.exp

      '''
      what is formula for backpropagation for 
      log(softmax(x)) with x a 2D tensor, 
      in particular why is there a summation over the out-gradient?
      '''
            
      ## ops_mathematics.summation. important to keep-dimensions...
      sum_grad = summation(out_grad, axes=1, keepdims=True)  # Sum along softmax axis
      # print( f'sum_grad = {sum_grad}')
      # print( f'sm_z = {sm_z}')
      ## the sum_grad term was suggested by chatGPT...
      ##  out_grad * I - sum_grad * softmax(z)
      ## ops from ops_mathematics via class Tensor and overloading of __mul__
      ## and __sub__ ....
      return out_grad - sm_z * sum_grad  # Apply (I - softmax) * grad_output
     

def logsoftmax(a):
    return LogSoftmax()(a)

import numpy as np


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes
        self.Z = None
        

    def compute(self, Z):
      self.Z = Z
      
      ## return array_api.squeeze( logsumexp_chat( Z, axis=self.axes))

      ## print( f'LogSumExp.compute: {type( Z)} {Z.shape} {Z}')
    
      max_z = array_api.max( Z, axis=self.axes, keepdims=True)
      ## max_z = array_api.squeeze( max_z)
      ## print( max_z)
      diff = array_api.subtract( Z, max_z)
      ## print( diff)
      exp_z = array_api.exp( diff)
      ## print( f'exp_z = {exp_z}')
      sum_z = array_api.sum( exp_z, axis=self.axes, keepdims=True)
      ## print( f'sum_z = {sum_z}')
      res = array_api.log( sum_z)
      ## print( f'log = {res}')
      res = array_api.add(  max_z, res)
      ## print( f'add = {res}')
      return array_api.squeeze( res)


    def gradient(self, out_grad, node):
      ## ChatGPT can show the gradient of grad_Z (LSE(Z)) = e^(Z - LSE(Z))...
      Z = node.inputs[0] ## self.Z
      if self.axes:
        ## array of ones...
        shape = [1] * len(Z.shape)
        j = 0
        ## the summation above reduces the shape along the given axes to be
        ## one. for the axes not in self.axes the shape is unchanged when
        ## computing log-sum-exp. and since we squeeze() the axes with
        ## shape == 1, we need to reconstitute the correct shape of the summed
        ## tensor, as the squeeze() discards axes of shape == 1...
        for i in range(len(shape)):
          ## axes can be an integer, a tuple or none...
          if ( type( self.axes) == tuple):
            if i not in self.axes:
              shape[i] = node.shape[j]
              j += 1
          elif ( type( self.axes) == int): ## integer
            if i != self.axes:
              shape[i] = node.shape[j]
              j += 1
          else:
            assert( False)

        ## shapes are either scalar or tuples...
        shape = tuple( shape)

        ## ops_mathematics via Tensor class...
        node_new = node.reshape(shape)
        grad_new = out_grad.reshape(shape)
      else:
        node_new = node
        grad_new = out_grad
      return grad_new * exp(Z - node_new)



def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

