"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """
    A special kind of tensor that represents parameters.
    Note that this is a sub-class of Tensor and thus use Parameter(xx) only
    if xx is a Tensor (not a tuple, a dict or a list) which are added to a class'
    dictionary as list, tuple, or dict.
    """


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        ''' return the list of modules in this module.'''
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        
        self.in_features = in_features ## Parameter( [in_features], dtype="float32")
        self.out_features = out_features ## Parameter( [out_features], dtype="float32")

        ## we have Tensor here..., not np.array...
        ## weight shape is (in_features, out_features)
        self.weight = init.kaiming_uniform( in_features, out_features, dtype="float32")
        ## fan_in == out_features != 1...
        self.bias = init.kaiming_uniform( out_features, 1, dtype="float32")
        ## bias shape is ( 1, out_features)
        self.bias = self.bias.reshape( (1, out_features))

        ## print( f"self.bias = {self.bias}")
        ## print( f"self.bias.shape = {self.bias.cached_data.shape}")

        ## finally record weight & bias as parameter Tensors..
        self.weight = Parameter( self.weight, dtype="float32")
        self.bias = Parameter( self.bias, dtype="float32")

        
    def forward(self, X: Tensor) -> Tensor:
      ## the output tensor is rows(X), out_features...
      ## the input X tensor should have shape N x in_features
      ## but sometimes it doesn't... hence we reshape the tensor (and do not squeeze the array)
      if ( X.cached_data.ndim > 2):
        ## print( f"X.dim > 2")
        x_shape = X.cached_data.shape
        XX = ops.reshape( X, (x_shape[1], x_shape[2]))
      else:
        XX = X
        x_shape = None

      ## now we're ready to use matmul()
      assert( XX.cached_data.ndim == 2)

      x_nbr_rows = XX.shape[0]
      x_nbr_cols = XX.shape[1]

      w_nbr_rows = self.weight.cached_data.shape[0]
      w_nbr_cols = self.weight.cached_data.shape[1]

      ## in the test runs, self.bias sometimes has the "wrong" shape and
      ## is (xx,) i.e. a vector
      ## print( f"self.bias = {self.bias.cached_data}")

      ## so we make sure that bias is a 2D tensor
      if ( len( self.bias.cached_data.shape) == 1):
        self.bias = ops.reshape( self.bias, (1, self.bias.cached_data.shape[0]))

      assert( len( self.bias.cached_data.shape) == 2)
      b_nbr_rows = self.bias.cached_data.shape[0]
      b_nbr_cols = self.bias.cached_data.shape[1]
      
      ## print( f"XX.shape = {XX.cached_data.shape}")
      ## print( f"weight.shape = {self.weight.cached_data.shape}")
      ## print( f"bias.shape = {self.bias.cached_data.shape}")
      
      assert( x_nbr_cols == w_nbr_rows)
      assert( w_nbr_cols == b_nbr_cols)
  
      ## here we need to explicitly broadcast self.bias which is a
      ## ndl Tensor and thus doesn't broadcast itself. see notice in HW..
      v1 = ops.matmul( XX, self.weight)
      bb = ops.broadcast_to( self.bias, v1.shape)
      v2 = ops.add( v1, bb)

      if ( x_shape is None):
        pass
      else:
        new_shape = ((x_shape[0], x_shape[1], b_nbr_cols))
        v2 = ops.reshape( v2, (new_shape))
         
      ## print( f"linear.forward: {v2}")

      return v2


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        ## reshape to 1st dim followed by all others except first
        return X.reshape((X.shape[0], -1))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
      return ops.relu( x)

class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules
        
    def forward(self, x: Tensor) -> Tensor:
        ## print( type( self.modules))
        ## modules is a tuple of modules...
        assert( len( self.modules) > 0)
        x_in = x
        x_out = None
       
        for m in self.modules: ## self._children():
          x_out = m(x_in)
          x_in = x_out

        return x_out
       


class SoftmaxLoss(Module):

    def forward(self, logits: Tensor, y: Tensor):
      """Return softmax loss.  Note that for the purposes of this assignment,
        you don't need to worry about "nicely" scaling the numerical properties
        of the log-sum-exp computation, but can just compute this directly.

        Args:
            Z (ndl.Tensor[np.float32]): 2D Tensor of shape
                (batch_size, num_classes), containing the non-logit predictions for
                each class. The normalization is done here!!!!
            y (ndl.Tensor[np.int32]): 1D Tensor of shape (batch_size)
                containing the class index

        Returns:
            Average softmax loss over the sample. (ndl.Tensor[np.float32])
      """

      def get_z_y( Z: Tensor, y: Tensor):
        ## numpy arrays
        ZZ = Z.cached_data
        yy = y.cached_data

        n_values = ZZ.shape[1]

        '''
        n_values = np.max(yy) + 1
        ## we can't be sure that max( yy) corresponds to the nbr of classes
        if ( n_values != ZZ.shape[1]):
          n_values = ZZ.shape[1]
          print( f'np.max = {np.max(yy)}')
          print( f'ZZ.shape = {ZZ.shape}')
          raise AssertionError
        '''

        one_hot_matrix = np.eye(n_values)[yy]
        ## select elements in colums by elementwise multiply

        ## print( f'n_values = {n_values}')
        ## print( f'ZZ.shape = {ZZ.shape}')
        ## print( f'one_hot_matrix.shape = {one_hot_matrix.shape}')

        m = ZZ * one_hot_matrix
        ## ignore columns with zero value by summing along rows
        m = np.sum(m, axis=1, keepdims=False)
        return Tensor( m)

        '''
        z_y = np.zeros( len( yy))
        for i in np.arange( len( yy)):
          z_y[i] = ZZ[i, yy[i]]

        print( f'z_y = {z_y}')

        return Tensor( z_y)
        '''
        

      ### BEGIN YOUR SOLUTION
      '''
        exp_sum = ops.logsumexp(logits, axes=(1, )).sum()
        z_y_sum = (logits * init.one_hot(logits.shape[1], y)).sum()
        return (exp_sum - z_y_sum) / logits.shape[0]
      '''
      
      one_hot_y = init.one_hot(logits.shape[1], y)
      len_y = logits.shape[0]
      ## TODO: figure out what happens when Tensor contains a scalar
      z_y = ops.summation(one_hot_y * logits / len_y)
      lse = ops.logsumexp(logits, (1,)) / len_y
      res = ops.summation( lse) - z_y
      return res 
      

          
      len_y = y.shape[0] ## scalar
      log_sum_exp = ops.logsumexp( logits, axes=1)  ## Tensor
      ## get the y-th element of the i-th row in z
      z_y = get_z_y( logits, y) ## Tensor...
      res = ops.summation(log_sum_exp - z_y)  
      res = ops.divide( res, Tensor( len_y))   
      return res
      

      ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim),requires_grad=True)
        self.bias = Parameter(init.zeros(dim),requires_grad=True)
        ## running_mean - the running mean used at evaluation time, elements initialized to 0.
        self.running_mean = init.zeros(self.dim)
        ## running_var - the running (unbiased) variance used at evaluation time, elements initialized to 1.
        self.running_var = init.ones(self.dim)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch_size = x.shape[0]
        feature_size = x.shape[1]
        # running estimates

        ## mean: sum along axes=0 i.e. columns...
        mean = x.sum(axes=(0,), keepdims=True) / batch_size
        x_minus_mean = x - mean.broadcast_to(x.shape)

        ## variance: again along rows axes, i.e. over columns
        var = (x_minus_mean ** 2).sum(axes=(0, ), keepdims=True) / batch_size

        if self.training:
            ##  compute the running estimates, using the equation given in notebook
            ## x_new = (1-momentum)* x_old + momentum * x_observed
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.data
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.data
            ## use the formula from notebook to compute y
            ## the denominator...
            x_std = ((var + self.eps) ** 0.5).broadcast_to(x.shape)
            ## the nominator is x-mean...
            normed = x_minus_mean / x_std
            res = normed * self.weight.broadcast_to(x.shape) + self.bias.broadcast_to(x.shape)
            ## print( f'res = {res}')
            return res
        else:
            ## training == False
            normed = (x - self.running_mean) / (self.running_var + self.eps) ** 0.5
            res = normed * self.weight.broadcast_to(x.shape) + self.bias.broadcast_to(x.shape)
            return res
        ### END YOUR SOLUTION



class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim),requires_grad=True)
        self.bias = Parameter(init.zeros(dim),requires_grad=True)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
      ### BEGIN YOUR SOLUTION
      mean = (x.sum((1,),keepdims=True)/x.shape[1]).reshape((x.shape[0], 1)).broadcast_to(x.shape)
      var = (((x - mean)**2).sum((1,),keepdims=True)/x.shape[1]).reshape((x.shape[0], 1)).broadcast_to(x.shape)
      deno = (var + self.eps)**0.5
      return self.weight.broadcast_to(x.shape) * (x - mean)/deno + self.bias.broadcast_to(x.shape)
      ### END YOUR SOLUTION

      ### BEGIN YOUR SOLUTION
      batch_size = x.shape[0]
      feature_size = x.shape[1]
      mean = x.sum(axes=(1, ),keepdims=True).reshape((batch_size, 1)) / feature_size
      x_minus_mean = x - mean.broadcast_to(x.shape)
      x_std = ((x_minus_mean ** 2).sum(axes=(1, ),keepdims=True).reshape((batch_size, 1)) / feature_size + self.eps) ** 0.5
      normed = x_minus_mean / x_std.broadcast_to(x.shape)
      return self.weight.broadcast_to(x.shape) * normed + self.bias.broadcast_to(x.shape)
      ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
          ## unpack the shape so that called func generates a tuple of ints and
          ## not a tuple of tuple...
          return x * (init.randb( *x.shape, p=(1 - self.p))) / (1- self.p)
        else:
          return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn( x) + x
        ### END YOUR SOLUTION
