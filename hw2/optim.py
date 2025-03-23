"""Optimization module"""
import needle as ndl
import numpy as np

from collections import defaultdict

class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        ## collections.defaultdict...
        self.u = defaultdict(float)
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION

        def clip_grad( grad, max_norm = 0.25):
          ''' 
          for numerical stability if the gradients become too small...
          based on ChatGPT suggestions which norm to use 
          '''
          
          ## frobenius_norm = np.linalg.norm(grad.cached_data, ord='fro')

          l2_norm = np.linalg.norm(grad.cached_data, ord=2)

          norm = l2_norm

          if ( norm > max_norm):
            grad = (max_norm/norm) * grad.cached_data
            return ndl.Tensor( grad, dtype='float32')
          else:
            return grad

        for i, param in enumerate(self.params):
            
            grad = ndl.Tensor(param.grad, dtype='float32').data       

            grad = grad + self.weight_decay * param.data     

            ## grad = clip_grad( grad)

            self.u[i] = self.momentum * self.u[i] + (1 - self.momentum) * grad

            param.data = param.data - self.u[i] * self.lr
        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        ## collections.defaultdict...
        self.m = defaultdict(float)
        self.v = defaultdict(float)

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        for i, param in enumerate(self.params):
            grad = ndl.Tensor(param.grad, dtype='float32').data + param.data * self.weight_decay
            # m_{t+1}, v{t+1}
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad**2
            # bias correction
            m_hat = (self.m[i]) / (1 - self.beta1 ** self.t)
            v_hat = (self.v[i]) / (1 - self.beta2 ** self.t)
            param.data = param.data - self.lr * m_hat / (v_hat ** 0.5 + self.eps) 
        ### END YOUR SOLUTION
