"""hw1/apps/simple_ml.py"""

import struct
import gzip
import numpy as np

import sys

sys.path.append("python/")
import needle as ndl


def parse_mnist(image_filename, label_filename):
    """Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR CODE
    
    import gzip
    import numpy as np
    import os
    import struct

    from urllib.request import urlretrieve 

    def load_data(src, num_samples):
        print("Downloading " + src)
        ## create a temporary file
        gzfname, h = urlretrieve(src, "./delete.me")
        print("Done.")
        ## unpack the data
        try:
            with gzip.open(gzfname) as gz:
                n = struct.unpack("I", gz.read(4))
                # Read magic number.
                if n[0] != 0x3080000:
                    raise Exception("Invalid file: unexpected magic number.")
                # Read number of entries.
                n = struct.unpack(">I", gz.read(4))[0]
                if n != num_samples:
                    raise Exception(
                        "Invalid file: expected {0} entries.".format(num_samples)
                    )
                ## number of rows & columns
                crow = struct.unpack(">I", gz.read(4))[0]
                ccol = struct.unpack(">I", gz.read(4))[0]
                if crow != 28 or ccol != 28:
                    raise Exception(
                        "Invalid file: expected 28 rows/cols per image."
                    )
                # Read data.
                res = np.frombuffer( gz.read(num_samples * crow * ccol), dtype=np.uint8)
        finally:
            ## delete the temp file
            os.remove(gzfname)
        ## reshape to (num_samples, crow * ccol) and normalize to [0.0..1.0]
        ## uint8 range is [0..255]...
        res = res.reshape((num_samples, crow * ccol)) / 255.0
        ## make sure it's float32 and not float64...
        return res.astype( 'float32')


    def load_labels(src, num_samples):
        print("Downloading " + src)
        gzfname, h = urlretrieve(src, "./delete.me")
        print("Done.")
        try:
            with gzip.open(gzfname) as gz:
                n = struct.unpack("I", gz.read(4))
                # Read magic number.
                if n[0] != 0x1080000:
                    raise Exception("Invalid file: unexpected magic number.")
                # Read number of entries.
                n = struct.unpack(">I", gz.read(4))
                if n[0] != num_samples:
                    raise Exception(
                        "Invalid file: expected {0} rows.".format(num_samples)
                    )
                # Read labels.
                res = np.frombuffer(gz.read(num_samples), dtype=np.uint8)
        finally:
            os.remove(gzfname)
        return res.reshape((num_samples))


    def try_download(data_source, label_source, num_samples):
        data = load_data(data_source, num_samples)
        labels = load_labels(label_source, num_samples)
        return data, labels
    

    ## server = 'https://yann.lecun.com/exdb/mnist/'
    server = 'https://raw.githubusercontent.com/fgnt/mnist/master/'
    
    # URLs for the train image and label data
    url_train_image = server + image_filename
    url_train_labels = server + label_filename
    num_train_samples = 60000

    print("Downloading train data: " + url_train_image + ", " + url_train_labels)
    train_features, train_labels = try_download(url_train_image, url_train_labels, num_train_samples)
    
    print( "Downloading done...")
    
    return ( train_features, train_labels)
  
    ### END YOUR CODE

def softmax_loss(Z_t, y_one_hot_t):
    """Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    ### BEGIN YOUR SOLUTION
    
    def find_idx_of_one( a, lower=0.95, upper=1.05):
      '''
        a: array of mainly zeros with a few values around 1.0
      '''
      b = np.nonzero( a)[0]

      for i in range( b.size):
        a_b_i = a[b[i]]
        if ( a_b_i >= lower and a_b_i <= upper):
          return b[i]

    ## from Needle Tensor to np.array...
    Z = Z_t.numpy()
    y = y_one_hot_t.numpy()

    ## the formulas for the gradients of Z and y are based on chatGPt
    ## for y, the gradient is all zeros
    ## for Z, we have dL/dZ = sigma - y and, since we are batching, we
    ## must use the average here...
    Z_grad = np.zeros( Z.shape)
    y_grad = np.zeros( y.shape)

    ## nbr of rows in Z == length of y?
    assert len(y) == np.shape( Z)[0]
   
    Z_exp = np.exp( Z)

    ## sum along exponentiated rows: sigma
    sum_rows_Z = np.sum( Z_exp, axis=1)
    ## take log(sigma)
    log_sum_rows_Z = np.log( sum_rows_Z)   
    
    ## the wonders of python indexing...: all rows, columns as per vector y
    ## res = - Z[:,y] + log_sum_rows_Z
    
    ## which doesn't seem to work: passing the array as index stalls python
    ## thus we use a dumb loop...
    res = log_sum_rows_Z
    for i in range( len( y)):
      ## y[i] here is a one-hot-vector of appropriate length
      ## we need to find the index of which the element is close to 1
      y_idx = find_idx_of_one( y[i])
      res[i] -= Z[i, y_idx]

      ## based on chatGPT... y_grad stays zero
      ## dL/dZ = sigma - y 
      Z_grad[i] = Z_exp[i]/sum_rows_Z[i]
      ## y_grad[i] = - np.log( Z_grad[i])
      Z_grad[i] = Z_grad[i] - y[i]

    ## set the gradients of the input args
    ## for Z, compute the average of the gradients...
    Z_t.grad = ndl.Tensor( Z_grad/len(y))
    y_one_hot_t.grad = ndl.Tensor( y_grad)

    ## len(y) == batch_size
    res = np.sum( res)/len( y)

    ## return data as Tensor
    return ndl.Tensor( res)

    ### END YOUR CODE

import math

def nn_epoch(X, y, W1, W2, lr=0.1, batch=100, debug=False):
    """Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    def normalize_Z( Z_t: ndl.Tensor):
      '''
      turns the values of Z into probabilities
      '''
      Z = Z_t.numpy()
      Z = np.exp( Z)
      sum_rows_Z = np.sum( Z, axis=1)  ## sum over rows
    
      Z_norm = np.zeros( Z.shape)  ## allocate S, the normalized Z

      for i in range( Z.shape[0]):  ## for all rows
        row_Z = Z[i, :]  ## alias for i-th row of Z
        Z_norm[i] = row_Z/sum_rows_Z[i]  ## normalize things by dividing each entry by the sum

        ## make sure we have a prob distribution...
        assert( math.isclose( np.sum( Z_norm[i]), 1.0, rel_tol=1E-6))

      Z_t.cached_data = Z_norm

      return Z_t

    def nn_batch( X, y, W_1, W_2, lr, batch_sz):
      '''Args:
          X (np.ndarray[np.float32]): 2D input array of size
              (batch_sz x input_dim).
          y (np.ndarray[np.uint8]): 2D class label array of size (batch_sz x num_classes)
          W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
              (input_dim, hidden_dim)
          W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
              (hidden_dim, num_classes)
          lr (float): step size (learning rate) for SGD
          batch_sz (int): size of SGD mini-batch
      '''

      assert( X.shape[1] == W1.numpy().shape[0])
      assert( X.shape[0] == batch_size)
      assert( len( y) == batch_size)
        
      # define some aliases, handling the tensor shapes...
      num_examples = X.shape[0]  ## nbr of rows in batch
      input_dim = X.shape[1]     ## size of input vectors
      hidden_dim = W1.numpy().shape[1]   ## size of hidden layer
      num_classes = W2.numpy().shape[1]  ## size of output
      
      X_t = ndl.Tensor( X)
      y_t = ndl.Tensor( y)

      X_W1 = ndl.matmul( X_t, W_1)
      Z_1 = ndl.relu( X_W1)
      assert( Z_1.numpy().shape == ( num_examples, hidden_dim))

      Z_1_W_2 = ndl.matmul( Z_1, W_2)  
     
      ## Z_1_W_2 = normalize_Z( Z_1_W_2)

      L = softmax_loss( Z_1_W_2, y_t)
      assert( Z_1_W_2.grad != None)

      ## get the gradients...
      Z_1_W_2.backward( Z_1_W_2.grad)

      ## print( f"Z_1_W_2.grad = {Z_1_W_2.grad}")

      grad_W_1 = W_1.grad
      grad_W_2 = W_2.grad

      grad_W_1 = ndl.mul_scalar( grad_W_1, -lr)
      grad_W_2 = ndl.mul_scalar( grad_W_2, -lr)

      W_1 = ndl.add( W_1, grad_W_1)
      W_2 = ndl.add( W_2, grad_W_2)
    
      return W_1.detach(), W_2.detach()

    ### iterate over samples, batch by batch...
    nbr_batches = math.ceil( len( y) / batch)
    num_classes = W2.numpy().shape[1] ## tensor to np array then shape
    batch_size = batch

    print( f'nbr_samples = {len(y)}, batch-size = {batch_size}, nbr batches = {nbr_batches}, num_classes = {num_classes}')

    for i in range( nbr_batches):

        print( f'batch = {i}')

        lb = i * batch_size
        ub = lb + batch_size
        
        ## if the nbr of samples is not an integer multiple of batch-size, we just stop
        if ( ub > len( y)):
            print( "truncated samples, last batch would exceed nbr of samples")
            break
            
        X_i = X[lb:ub, :]
        y_i = y[lb:ub]

        ## one hot encoded
        y_h_i = np.zeros( (len( y_i), num_classes))
        y_h_i[np.arange(len( y_i)), y_i] = 1

        W1, W2 = nn_batch( X_i, y_h_i, W1, W2, lr, batch_size)
   
    return ( W1, W2)

### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


def loss_err(h, y):
    """Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
