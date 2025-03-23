/* this is the C version of the dlsys HW0 running a neural net without
   using autograd differentiation.
   it contains the linear regression code as well as the 2 layer neural net
   using explicitly generated gradients
*/

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#include "tensor.h"
#include "mnist.h"

#define ABS(x) (x<0?-x:x)

#define TRUE 1
#define FALSE 0

/**
 * True if the values a and b are close to each other and False otherwise.
 *
 * @param a
 * @param b
 * @param rel_tol relative tolerance default 1E-9
 * @param abs_tol absolute tolerance for test on 0. must be >= 0, default 0.
 */
unsigned char isclose( const double a, const double b, const double rel_tol, const double abs_tol) {
  assert( abs_tol >= 0 && rel_tol >= 0);
  const double diff = ABS(a-b);
  const double rel_diff = rel_tol * fmax( ABS(a), ABS(b));
  const double max_diff = fmax( rel_diff, abs_tol);
  return (diff < max_diff?1:0);
}


/**
  given a 1D tensor of categorical labels, genrate a 2D tensor of one-hot vectors

  v: 1D tensor of categorical labels

  returns 2D tensor of one-hot row vectors, dtype = T_INT8
 */
static t_tensor gen_one_hot_rows( const t_tensor v) {

  assert( t_is1D( v));

  t_tensor mx = t_max( v, NULL, FALSE);
  t_tensor mi = t_min( v, NULL, FALSE);
  const int64_t nbr_categories = (int64_t) (t_scalar( mx) - t_scalar( mi) + 1);

  if ( nbr_categories <= 0) {
    fprintf( stderr, "gen_one_hot_rows: no categories?\n");
  }

  // constant one as t_value
  t_value one;
  one.dtype = T_INT8;
  one.u.c = (uint8_t) 1;

  // square matrix
  uint32_t shape[2] = { v->shape[0], nbr_categories};
  
  // vectors of 0,1: one-hot-vector...
  t_tensor c = t_new( 2, shape, T_INT8);
  
  for ( uint32_t i = 0; i < shape[0]; i++) {
    // v is 1D
    t_value tv;
    assign_to_value( &tv, v, i, T_INT32);

    t_2D_set_tv( c, i, tv.u.l, one);
  }

  T_FREE( mx);
  T_FREE( mi);

#if 0
  // check that all data is 0 or 1
  for ( uint32_t i = 0; i < t_size( c); i++) {
    const int8_t *pp = (int8_t *) c->data;
    const int8_t tt = pp[i];
    assert( tt == 0 || tt == 1);
  }
#endif

#if 0
  fprintf( stderr, "I_y: %d [%d, %d]\n",
	   c->rank, SHAPE_DIM( c, 0), SHAPE_DIM( c, 1));
#endif
  
  return c;
}

static t_tensor normalize( const t_tensor Z) {

  // assert( !t_has_nan( Z));
  // assert( !t_has_inf( Z));
  
  uint8_t row_axis[2] = {0, 1};
 t_tensor sum_rows_Z = t_sum( Z, row_axis, FALSE);  // sum over rows
  // assert( !t_has_nan( sum_rows_Z));
  // assert( !t_has_inf( sum_rows_Z));

  // fprintf( stderr, "sum_rows_Z: %d [%d]\n", sum_rows_Z->rank, sum_rows_Z->shape[0]);

  assert( SHAPE_DIM( Z, 0) == SHAPE_DIM( sum_rows_Z, 0));
  assert( Z->rank == 2);

  t_tensor Z_norm = t_clone( Z); // copy values

  for ( unsigned int i = 0; i < SHAPE_DIM( Z, 0); i++) {

    const double srz_i = t_1D_get( sum_rows_Z, i);
    // assert( srz_i != 0.0);

    double sum = 0;
    
    for ( unsigned int j = 0; j < SHAPE_DIM( Z, 1); j++) {
      double zn = t_2D_get( Z_norm, i, j);

      zn /= srz_i;
      // assert( !isnan( sum));

      sum += zn;
      // assert( !isnan( sum));
      
      t_2D_set( Z_norm, i, j, zn);
    }

    if ( ! isclose( sum, 1.0, 1E-5, 1E-5)) {
      fprintf( stderr, "sum not close to 1.0: %f\n", sum);
      exit( -1);
    }
  }
  
  t_free( sum_rows_Z);
  
  return Z_norm;
}

static void softmax_regression_batch( const t_tensor X,
				      const t_tensor y,
				      t_tensor theta,
				      const t_tensor lr_over_batch_size) {

  fprintf( stderr, "X: %d [%d, %d], y: %d [%d]\n",
	   X->rank, X->shape[0], X->shape[1],
	   y->rank, y->shape[0]);

  // assert( !t_has_inf( X));
  // assert( !t_has_inf( theta));
  
  t_tensor Z = t_matmul( X, theta, NULL);  // matrix multiply
  // assert( !t_has_inf( Z));
  t_tensor Z_exp = t_exp( Z, NULL);        // exponentiate
  // assert( !t_has_inf( Z_exp));


  t_tensor I_y = gen_one_hot_rows( y);
  t_tensor Z_norm = normalize( Z_exp);

  // Z_minus_I = Z_norm - I_y
  t_tensor Z_minus_I = t_subtract( Z_norm, I_y, NULL);

  t_tensor X_t = t_transpose( X, NULL);

  t_tensor grad = t_matmul( X_t, Z_minus_I, NULL);

  // grad = (lr/batch_size) * grad, in-situ
  grad = t_multiply( grad, lr_over_batch_size, grad);
  // assert( !t_has_nan( grad));
  
  assert( t_same_shape( grad, theta));

  // in-situ subtraction
  theta = t_subtract( theta, grad, theta);
  // assert( !t_has_nan( theta));

  t_free( Z);
  t_free( Z_exp);
  t_free( Z_norm);
  t_free( I_y);
  t_free( Z_minus_I);
  t_free( X_t);
  t_free( grad);
	   
}

static void set_slice_idx( uint32_t s[][2], const unsigned int idx,
			   const unsigned int lb, const unsigned int ub) {
  s[idx][0] = lb;
  s[idx][1] = ub;
}

static void softmax_regression( const t_tensor X,
				const t_tensor y,
				const t_tensor theta,
				const double lr,
				const unsigned int batch_size) {

  assert( t_rank( X) == 2);
  assert( t_rank( y) == 1);

  const unsigned int y_len = t_1D_len( y);
  const unsigned int nbr_batches = (unsigned int) ceil( y_len/batch_size);

  fprintf( stderr, "nbr samples: %d, nbr_batches: %d, batch_size: %d\n",
	   y_len, nbr_batches, batch_size);

  // to slice the input data into batches...
  uint32_t X_i_idx[2][2];
  uint32_t Y_i_idx[1][2];

  t_tensor lr_over_batch_size = t_new_scalar( (double) lr/batch_size, T_FLOAT);
  
  for ( unsigned int i = 0 ; i < nbr_batches; i++) {
    const unsigned int lb = i * batch_size;
    const unsigned int ub = lb + batch_size;
        
    // if the nbr of samples is not an integer multiple of batch-size, we just stop
    if ( ub > y_len) {
      fprintf( stderr, "truncated samples, last batch would exceed nbr of samples");
      break;
    }

    set_slice_idx( Y_i_idx, 0, lb, ub);

    set_slice_idx( X_i_idx, 0, lb, ub);
    set_slice_idx( X_i_idx, 1, 0, X->shape[1]);

#if 0
    fprintf( stderr, "X: [%d:%d, %d:%d], y: [%d:%d]\n",
	     X_i_idx[0][0], X_i_idx[0][1], X_i_idx[1][0], X_i_idx[1][1],
	     Y_i_idx[0][0], Y_i_idx[0][1]);
#endif
    fprintf( stderr, "[%d : %d]\n", lb, ub);

    t_tensor X_i = t_slice( X, X_i_idx, 2);  // X[lb:ub, :];
    t_tensor y_i = t_slice( y, Y_i_idx, 1);  // y[lb:ub];

    softmax_regression_batch( X_i, y_i, theta, lr_over_batch_size);

    t_free( X_i);
    t_free( y_i);
  }

  t_free( lr_over_batch_size);
  
}

static void nn_batch( const t_tensor X,
		      const t_tensor y,
		      t_tensor W_1,
		      t_tensor W_2,
		      const t_tensor lr_over_batch_size) {

  const uint32_t num_examples = SHAPE_DIM( X, 0);    // nbr of rows in batch
  const uint32_t input_dim    = SHAPE_DIM( X, 1);    // size of input vectors
  const uint32_t hidden_dim   = SHAPE_DIM( W_1, 1);   // size of hidden layer
  const uint32_t num_classes  = SHAPE_DIM( W_2, 1);   // size of output

  // assert( X_shape.shape[0] == y_shape.shape[0]);  
  assert( SHAPE_DIM( X, 0) == SHAPE_DIM( y, 0));     // nbr of rows, input & output
  assert( SHAPE_DIM( X, 1) == SHAPE_DIM( W_1, 0));   // nbr of cols input == nbr of rows W1
  assert( SHAPE_DIM( W_1, 1) == SHAPE_DIM( W_2, 0)); // nbr of cols W1 == nbr of rows W2

#if 0
  fprintf( stderr, "num_examples: %d input_dim: %d, hidden_dim: %d, num_classes: %d\n",
	   num_examples, input_dim, hidden_dim, num_classes);
#endif
  
  /*
    X_W1 = np.matmul( X, W_1)
    Z_1 = ReLU( X_W1)
    assert( Z_1.shape == ( num_examples, hidden_dim))
  */
  t_tensor X_W1 = t_matmul( X, W_1, NULL);
  t_tensor Z_1 = t_relu( X_W1, NULL);

  assert( t_check_shape( Z_1, num_examples, hidden_dim, -1));

  /*
    ## multiply Z_1, W_2, exponentiate result and normalize
    Z_1_W_2 = normalize_Z( np.exp( np.matmul( Z_1, W_2)))
    I_y = gen_one_hot_y( y, num_classes)
    assert( I_y.shape == Z_1_W_2.shape)
  */
  t_tensor Z1_W2 = t_matmul( Z_1, W_2, NULL);
  // assert( !t_has_nan( Z1_W2));
  // assert( !t_has_inf( Z1_W2));

  // t_dump( Z1_W2, TRUE);
  
  Z1_W2 = t_exp( Z1_W2, Z1_W2); // in-situ
  // assert( !t_has_nan( Z1_W2));
  // assert( !t_has_inf( Z1_W2));

  t_tensor Z1_W2_norm = normalize( Z1_W2);

  t_tensor I_y = gen_one_hot_rows( y);

  assert( t_same_shape( I_y, Z1_W2_norm));

  /*
    G_2 = Z_1_W_2 - I_y  
    ## elementwise multiply of the ReLU_derivative of Z_1 with (G_2 * W_2^T)
    G_1 = np.multiply( ReLU_derivative( Z_1), np.matmul( G_2, np.transpose( W_2)))
    
    assert( G_2.shape == ( num_examples, num_classes))
    assert( G_1.shape == ( num_examples, hidden_dim))
  */
  t_tensor G2 = t_subtract( Z1_W2_norm, I_y, NULL);

  t_tensor W2_t = t_transpose( W_2, NULL);
  t_tensor G2_W2t = t_matmul( G2, W2_t, NULL);
  t_tensor Z1_derivative = t_relu_deriv( Z_1, NULL);

  // elementwise multiplication...
  t_tensor G1 = t_multiply( Z1_derivative, G2_W2t, NULL);

  t_free( W2_t);
  t_free( G2_W2t);
  t_free( Z1_derivative);

  // check shapes
  assert( t_check_shape( G1, num_examples, hidden_dim, -1));
  assert( t_check_shape( G2, num_examples, num_classes, -1));

  /*
    grad_W_1 = np.matmul( np.transpose( X), G_1)/batch_size
    grad_W_2 = np.matmul( np.transpose( Z_1), G_2)/batch_size

    W_1 -= lr * grad_W_1
    W_2 -= lr * grad_W_2
  */
  t_tensor X_t = t_transpose( X, NULL);
  t_tensor Z1_t = t_transpose( Z_1, NULL);

  t_tensor grad_W1 = t_matmul( X_t, G1, NULL);
  t_tensor grad_W2 = t_matmul( Z1_t, G2, NULL);

  grad_W1 = t_multiply( grad_W1, lr_over_batch_size, grad_W1); // in-situ
  grad_W2 = t_multiply( grad_W2, lr_over_batch_size, grad_W2); // in-situ

  assert( t_same_shape( grad_W1, W_1));
  assert( t_same_shape( grad_W2, W_2));
	  
  W_1 = t_subtract( W_1, grad_W1, W_1); // element-wise, in-situ
  W_2 = t_subtract( W_2, grad_W2, W_2); // element-wise, in-situ
  
  t_free( X_t);
  t_free( Z1_t);
  t_free( grad_W1);
  t_free( grad_W2);


  t_free( X_W1);
  t_free( Z_1);
  t_free( Z1_W2);
  t_free( Z1_W2_norm);
  t_free( I_y);
  t_free( G2);
  
}


static void nn_epoch( const t_tensor X,
		      const t_tensor y,
		      const t_tensor W_1,
		      const t_tensor W_2,
		      const double lr,
		      const unsigned int batch_size,
		      const unsigned int hidden_dim) {

  assert( t_rank( X) == 2);
  assert( t_rank( y) == 1);

  const unsigned int y_len = t_1D_len( y);
  const unsigned int nbr_batches = (unsigned int) ceil( y_len/batch_size);

  fprintf( stderr, "nbr samples: %d, nbr_batches: %d, batch_size; %d\n", y_len, nbr_batches, batch_size);

  // to slice the input data into batches...
  uint32_t X_i_idx[2][2];
  uint32_t Y_i_idx[1][2];

  t_tensor lr_over_batch_size = t_new_scalar( (double) lr/batch_size, T_FLOAT);
  
  for ( unsigned int i = 0 ; i < nbr_batches; i++) {
    const unsigned int lb = i * batch_size;
    const unsigned int ub = lb + batch_size;
        
    // if the nbr of samples is not an integer multiple of batch-size, we just stop
    if ( ub > y_len) {
      fprintf( stderr, "truncated samples, last batch would exceed nbr of samples");
      break;
    }

    set_slice_idx( Y_i_idx, 0, lb, ub);

    set_slice_idx( X_i_idx, 0, lb, ub);
    set_slice_idx( X_i_idx, 1, 0, X->shape[1]);

#if 0
    fprintf( stderr, "X: [%d:%d, %d:%d], y: [%d:%d]\n",
	     X_i_idx[0][0], X_i_idx[0][1], X_i_idx[1][0], X_i_idx[1][1],
	     Y_i_idx[0][0], Y_i_idx[0][1]);
#endif
    fprintf( stderr, "[%d : %d]\n", lb, ub);
    
    t_tensor X_i = t_slice( X, X_i_idx, 2);  // X[lb:ub, :];
    t_tensor y_i = t_slice( y, Y_i_idx, 1);  // y[lb:ub];

    nn_batch( X_i, y_i, W_1, W_2, lr_over_batch_size);

    t_free( X_i);
    t_free( y_i);
  }

  t_free( lr_over_batch_size);
  
}


static unsigned int get_y_max( const t_tensor y) {
  t_tensor y_max = t_max( y, NULL, FALSE);
  const unsigned int m = (unsigned int) t_scalar( y_max);
  t_free( y_max);
  return m;
}

// from 2D into 1D
t_tensor flatten_images( t_tensor t) {
  // t_shape_struct t_shape;
  // t_get_shape( t, &t_shape);
  
  assert( t_rank( t) == 3);

  uint32_t shape[2];
  shape[0] = SHAPE_DIM( t, 0);  // nbr of images
  shape[1] = SHAPE_DIM( t, 1) * SHAPE_DIM( t, 2); // rows * cols
  
  t_tensor tt = t_reshape( t, 2, shape, t);
  assert( t_rank( tt) == 2);
  return tt;
}

void main( int argc, char **argv) {

  // read the data
  t_tensor mnist_images = mnist_get_images( MNIST_TRAIN_IMAGES, TRUE);
  t_tensor mnist_labels = mnist_get_labels( MNIST_TRAIN_LABELS);

  mnist_images = flatten_images( mnist_images);
  
  assert( t_rank( mnist_images) == 2);
  assert( t_rank( mnist_labels) == 1);

  unsigned int y_max = get_y_max( mnist_labels);
    
  const double lr = 0.1;
  const unsigned int batch_size = 100;

  const unsigned char use_softmax_regression = FALSE;

  if ( use_softmax_regression) {
    // allocate a zero filed 2D tensor of FLOAT

    // theta = np.zeros((X.shape[1], y.max()+1), dtype=np.float32)

    uint32_t theta_shape[2];
    theta_shape[0] = SHAPE_DIM( mnist_images, 1);  // size of images: 28 x 28
    theta_shape[1] = y_max+1;           // max value of label

    t_tensor theta = t_new( 2, theta_shape, T_FLOAT);
  
    softmax_regression( mnist_images, mnist_labels, theta, lr, batch_size);

    t_free( theta);

  } else {

    const uint32_t hidden_dim = 500;
    
    uint32_t W1_shape[2];
    W1_shape[0] = SHAPE_DIM( mnist_images, 1);  // size of images: 28 x 28
    W1_shape[1] = hidden_dim;        // dimension of hidden layer

    uint32_t W2_shape[2];
    W2_shape[0] = hidden_dim;        // size of images: 28 x 28
    W2_shape[1] = y_max+1;           // max value of label

    /*
      W1 = np.random.randn(n, hidden_dim).astype(np.float32) / np.sqrt(hidden_dim)
      W2 = np.random.randn(hidden_dim, k).astype(np.float32) / np.sqrt(k)
    */

    // create a normally distributed random 2D tensor, mean 0, variance 1
    t_tensor W_1 = t_randn( 2, W1_shape, T_FLOAT);
    t_tensor W_2 = t_randn( 2, W2_shape, T_FLOAT);

    // scale the weight matrices... to avoid overflows in exponentiation
    W_1 = t_div_scalar( W_1, sqrt( (double) hidden_dim));
    W_2 = t_div_scalar( W_2, sqrt( (double) (y_max+1)));

    nn_epoch( mnist_images, mnist_labels, W_1, W_2, lr, batch_size, hidden_dim);

    t_free( W_1);
    t_free( W_2);
  }

  t_free( mnist_images);
  t_free( mnist_labels);
  
  exit( 0);
  
}
