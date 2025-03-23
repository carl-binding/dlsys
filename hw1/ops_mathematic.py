"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

BACKEND = "np"
import numpy as array_api

class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power. a^b """

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return array_api.power( a, b)
                
    def gradient(self, out_grad, node):
        a = node.inputs[0]
        b = node.inputs[1]
        # Compute the gradient using the chain rules
        ## d(a^b)/da = b * a^(b-1), d(a^b)/db = a^b * ln( a)
        grad_a = out_grad * b * power( a, b-1)
        grad_b = out_grad * node * log( a)
        return grad_a, grad_b


def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return array_api.power( a, self.scalar)
        

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION// v = x^s, dv/dx = s*x^(s-1)
        ## v = x^s, dv/dx = s*x^(s-1)
        x = node.inputs[0]
        v1 = power_scalar( x, self.scalar-1)
        v2 = mul_scalar( v1, self.scalar)
        return multiply( out_grad, v2)
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
      return array_api.divide( a, b)

    def gradient(self, out_grad, node):
      a, b = node.inputs
      
      r1 = divide( out_grad, b)

      b_squared = multiply( b, b)
      neg_b_squared = negate( b_squared)
      t1 = divide(a, neg_b_squared)
      r2 = multiply( out_grad, t1)

      return r1, r2


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
      return array_api.divide( a, self.scalar)

    def gradient(self, out_grad, node):
      ## v = a/s
      ## a = node.inputs[0]
      ## dv/da = 1/s
      return divide_scalar( out_grad, self.scalar)
      
def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        ## note: numpy.transpose requires correct nbr of axes... see the docs
        ## axes must contain a permutation of dim-1 values...
        return array_api.transpose( a, axes=self.axes)
        ## raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ## A_bar = inv_transpose( C_bar), i.e. grad_in_node = inv_transpose( out_grad)
        ## and the inverse of the tranpose is the tranpose itself...
        return ( transpose( out_grad, self.axes))
      


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape( a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
      ## A_bar = inv_reshape( C_bar), i.e. grad_in_node = inv_reshape( out_grad)
      ## and the inverse of the reshape is the reshape to the original shape...
      ### BEGIN YOUR SOLUTION
      orig_shape = node.inputs[0].shape
      return reshape( out_grad, orig_shape)
      ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.broadcast_to( a, self.shape)
        ## raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        out_shape = out_grad.shape
        in_shape = node.inputs[0].shape
        ## special case for scalars and tensors of dim 0
        if ( len( in_shape) == 0):
          return [ summation( out_grad)]
        if ( len( in_shape) == 1):
          return [ summation( out_grad)]
        # Determine the broadcasted axes
        if ( len( out_shape) != len( in_shape)):
          raise ValueError( "BroadcastTo.gradient: mismatching shapes")
        ## we collect the axis which have been broadcast. assuming the original shape was 1...
        ## which is the case for the tests, but may not be the case in general
        axes_to_sum = [i for i, (b, o) in enumerate(zip(out_shape, in_shape)) if o == 1]
        print( axes_to_sum)
    
        # Sum over broadcasted axes
        grad_input = summation( out_grad, axes=tuple(axes_to_sum)) ## , keepdims=True)
    
        # Reshape back to the original shape
        return grad_input.reshape(in_shape)
        


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
      ## this does keep dimensions. and thus fails a test. the
      ## cure is to use numpy.squeeze which howver causes another test to fail
        return array_api.sum( a, axis = self.axes, keepdims=True)

    def gradient(self, out_grad, node):
      input = node.inputs[0]
      # Broadcast out_grad to the input shape, i.e. the
      # smaller out-grad tensor to the larger input shape.
      ## courtesy of ChatGPT
      grad_input = broadcast_to(out_grad, input.shape)
      return grad_input



def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        return array_api.matmul( a, b)

    def gradient(self, out_grad, node):
      a, b = node.inputs
      ## v = a * b, 
      ## https://robotchinwag.com/posts/gradient-of-matrix-multiplicationin-deep-learning/
      a_t = transpose(a)
      b_t = transpose(b)
      r1 = matmul( out_grad, b_t)
      r2 = matmul( a_t, out_grad)
      return r1, r2


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.negative( a)
        ## raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        return negate( out_grad)


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log( a)
        ## raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        a = node.inputs[0]
        return divide( out_grad, a)


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        return array_api.exp( a)
        
    def gradient(self, out_grad, node):
        exp_a = node
        ## chain rule: h(x) = g(f(x)) => h'(x) = f'(x)*g'(f(x))
        ## f'(x) is the out_grad, f(x) == exp_a, g'( exp()) = exp()
        return multiply( out_grad, exp_a)


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        return array_api.maximum( a, 0)

    def gradient(self, out_grad, node):
      ## type of cached_data is NDArray 
      d = node.realize_cached_data()
      ## derivative of ReLU is 1 or 0. type is NDArray
      gd = ( d > 0) * 1

      ## partial_adjoint * out_gradient... considering the fucked up types
      gd = array_api.multiply( gd, out_grad.numpy())

      ## and make a Tensor out of the NDArray...
      return Tensor( gd)

def relu(a):
    return ReLU()(a)

