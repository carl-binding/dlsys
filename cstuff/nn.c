#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <float.h>
#include <stdarg.h>

#include "tensor.h"
#include "nn.h"

#define FALSE 0
#define TRUE 1

#define LOG_DOTS_PER_LINE 10

void nn_log_progress( const char *label,
		      const uint32_t m,
		      const uint32_t max_len) {
  if ( (m % (LOG_DOTS_PER_LINE * max_len)) == 0) {
    fprintf( stderr, "%s [%d]:  ", label, m);
  }
  if ( ((m+1) % max_len) == 0) {
    fprintf( stderr, ". ");
    if ( ((m+1) % (LOG_DOTS_PER_LINE * max_len)) == 0)
      fprintf( stderr, "\n");
  }
}

t_tensor nn_normalize( t_tensor t,
		       const double origin,
		       const double spread,
		       double min_of_t,
		       double max_of_t) {
  assert( t_assert( t));
  assert( spread > 0.0);
  
  if ( t->dtype != T_FLOAT && t->dtype != T_DOUBLE) {
    fprintf( stderr, "nn_normalize: input data must be double or float\n");
    return NULL;
  }

  // find min and max over t unless they are known...
  t_tensor tmax = NULL;
  t_tensor tmin = NULL;
  
  if ( max_of_t == DBL_MIN) { // search for max
    tmax = t_max( t, NULL);
  } else {
    tmax = t_new_scalar( max_of_t, t->dtype);
  }
  if ( min_of_t == DBL_MAX) { // search for min
    tmin = t_min( t, NULL);
  } else {
    tmin = t_new_scalar( min_of_t, t->dtype);
  }

  // compute max - min
  t_tensor delta = t_subtract( tmax, tmin, NULL);

  if ( t_scalar( tmin) != 0) {
    // shift data towards origin by -min
    t = t_subtract( t, tmin, t);  // shift to 0.
  } else {
    // nothing
  }
  
  if ( spread != 1.0) {
    t_tensor scale = t_new_scalar( spread, t->dtype);
    scale = t_divide( scale, delta, scale);
    t = t_multiply( t, scale, t);
    T_FREE( scale);
  } else {
    // scale data to be [0..1]
    t = t_divide( t, delta, t);
  }

  if ( origin != 0.0) { // shift data towards new origin
    t_tensor o = t_new_scalar( origin, t->dtype);
    t = t_add( t, o, t);
    T_FREE( o);
  }

  // release garbage
  T_FREE( tmax);
  T_FREE( tmin);
  T_FREE( delta);
  
  return t;
}


t_tensor nn_categorical( const t_tensor v) {

  assert( t_is1D( v));

  t_tensor mx = t_max( v, NULL);
  t_tensor mi = t_min( v, NULL);
  const int64_t nbr_categories = (int64_t) (t_scalar( mx) - t_scalar( mi) + 1);

  if ( nbr_categories <= 0) {
    fprintf( stderr, "nn_categorical: no categories?\n");
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
  
  return c;
}

t_tensor nn_arr_to_categorical( const uint16_t len, const uint32_t categories[]) {
  assert( len > 0 && categories != NULL);
  const uint32_t shape[1] = { len};

  // we throw all categories into a 1D tensor...
  t_tensor tc = t_new( 1, shape, T_INT32);
  for ( uint32_t i = 0; i < len; i++) {
    t_value tv;

    tv.dtype = T_INT32;
    tv.u.l = categories[i];

    if ( tv.u.l < 0) {
      fprintf( stderr, "nn_categorical: category cannot be negative\n");
      T_FREE( tc);
      return NULL;
    }

    assign_to_tensor( tc, i, tv, FALSE);
  }
  
  // t_dump( tc, TRUE);
  
  t_tensor cat = nn_categorical( tc);
  
  T_FREE( tc);
  return cat;
}

#if 0

double sigmoid( const double x) {
  return ( 1.0 / ( 1.0 + exp( -x)));
}

double sigmoid_derivate( const double x) {
  const double s_x = sigmoid( x);
  return s_x * ( 1.0 - s_x);
}

double tanh_derivate( const double x) {
  const double cosh_x = cosh( x);
  return ( 1.0/(cosh_x * cosh_x));
}

double relu( const double x) {
  return (x > 0)?x:0;
}

double relu_derivate( const double x) {
  return (x > 0)?1:0;
}

#endif

// function over a 1 D array, returning a 1 D array.
// if out != NULL, in-situ use of out
t_tensor softmax( const t_tensor a, t_tensor out) {

  assert( t_assert( a));
  assert( t_is1D( a));
  
  const unsigned len = t_1D_len( a);

  const unsigned long shape[1] = {len};

  t_tensor mx = t_max( a, NULL);
  assert( t_is0D( mx));
  
  // allocate output if needed
  if ( out == NULL) {
    out = t_new_vector( len, a->dtype, NULL);
  } else {
    // check if output has proper size
    assert( t_assert( out));
    assert( t_is1D( out));
    assert( a != out);
    
    if ( t_1D_len( out) != len) {
      fprintf( stderr, "softmax: output tensor length doesn't match");
      return NULL;
    }
    if ( out->dtype != a->dtype) {
      fprintf( stderr, "softmax: output tensor not of same type\n");
      return NULL;
    }
  }
  
  // a - mx, over array a
  out = t_subtract( a, mx, out);
  
  // exp( a - mx) over array c
  out = t_apply( out, exp, out);
  
  // sum over exponentiated array
  t_tensor s = t_sum( out, NULL);
  assert( t_is0D( s));

  // element-wise division
  assert( t_is1D( out));
  out = t_divide( out, s, out);

  T_FREE( s);
  T_FREE( mx);
  
  return out;
}

t_tensor softmax_derivate( const t_tensor a) {

  assert( t_assert( a));
  assert( t_is1D( a));

  if ( a->dtype != T_DOUBLE && a->dtype != T_FLOAT) {
    fprintf( stderr, "softmax_derivate: tensor is not T_DOUBLE or T_FLOAT\n");
    return NULL;
  }
  
  t_tensor p = softmax( a, NULL);
  assert( t_is1D( p));

  const unsigned long n = t_1D_len( p);
  t_tensor pd = t_new_matrix( n, n, a->dtype, NULL);

  for ( unsigned long i = 0; i < n; i++) {
    t_value p_i;
    assign_to_double( &p_i, p, i);

    for ( unsigned long j = 0; j < n; j++) {
      t_value p_j;
      assign_to_double( &p_j, p, j);
      
      t_value tv;
      tv.dtype = a->dtype;
      if ( i == j) {
	// dy[i,j] = p_i * ( 1 - p_j)
	tv.u.d = p_i.u.d * ( 1 - p_j.u.d);
      } else {
	// dy[i,j] = -p_i * p_j
	tv.u.d = - p_i.u.d * p_j.u.d;
      }
      const unsigned long long off = i * n + j;
      // coerce as needed
      assign_to_tensor( pd, off, tv, TRUE);
    }
  }

  return pd;
  
}

static uint32_t get_bit_pos( const t_tensor t) {
  
  assert( t_assert( t));
  assert( t_is1D( t));
  assert( t->dtype == T_INT8);

  // could use t_argmax(), but this is faster...
  
  // we assume that one hot vector is encoded as uint8_t
  uint8_t *p = (uint8_t *) t->data;
  for ( uint64_t i = 0; i < t_size( t); i++) {
    if ( *p++ == 1) {
      return i;
    }
  }
  // we assume that at least one bit is set to 1
  assert( FALSE);
}

double nn_cost( const nn_network nn,
		const t_tensor y_hat,
		const t_tensor t) {
  assert( nn != NULL);
  
  assert( t_assert( y_hat));
  assert( t_assert( t));


  assert(( t_is1D( t) || t_is0D( t)) && t->dtype == T_INT8);
  // output is floating point...
  assert(( t_is1D( y_hat) || t_is0D( t)) && y_hat->dtype == NN_DTYPE);

  assert(( t_is0D(t) && t_is0D( y_hat)) ||
	 (t_1D_len( t) == t_1D_len( y_hat)));

  switch ( nn->cost_func) {
  case NN_MSE:
    {
      // y_hat - t
      t_tensor diff = t_subtract( y_hat, t, NULL);
      const double norm = t_norm( diff, T_ORD_FROBENIUS);
      T_FREE( diff);
      return norm * norm * 0.5;
    }
    break;
  case NN_CROSS_ENTROPY:
    // return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))
    // element wise multiplication of arrays...
    {
      // a is output of neural net == y_hat
      // y is truth == t
      t_tensor a = y_hat;
      t_tensor y = t;
      
      t_tensor log_a = t_log( a, NULL);
      assert( t_assert( log_a));
      t_tensor one_minus_a = t_subtract( t_one(), a, NULL);
      assert( t_assert( one_minus_a));
      t_tensor one_minus_y = t_subtract( t_one(), y, NULL);
      assert( t_assert( one_minus_y));
      t_tensor log_one_minus_a = t_log( one_minus_a, NULL);
      assert( t_assert( log_one_minus_a));

      // (1 - y) * log( 1 - a)
      t_tensor accu = t_multiply( one_minus_y, log_one_minus_a, NULL);
      assert( t_assert( accu));

      // -y * np.log( a)
      t_tensor term = t_multiply( y, log_a, NULL);
      assert( t_assert( term));
      term = t_multiply( term, t_minus_one(), term);
      assert( t_assert( term));      
      
      // - y * log( a) - (1-y) * log( 1 - a)
      accu = t_subtract( term, accu, accu);
      assert( t_assert( accu));
      accu = t_nan_to_num( accu, FALSE, 0.0, DBL_MAX, DBL_MIN);
      assert( t_assert( accu));

      t_tensor cost = t_sum( accu, NULL);
      assert( t_is0D( cost));

      T_FREE( log_a);
      T_FREE( one_minus_a);
      T_FREE( one_minus_y);
      T_FREE( log_one_minus_a);
      T_FREE( accu);
      T_FREE( term);

      double cc = t_scalar( cost);
      T_FREE( cost);
      
      return cc;
      
    }
    break;
  case NN_LOG_LIKELIHOOD:
    // C = - log( activation of truth label)
    // Nielsen (80)
    
    // need to figure out which bit in t is hot and use it as index into y_hat
    {
      const uint32_t label_idx = get_bit_pos( t);
      const double act_label = t_1D_get( y_hat, label_idx);

      return -log( act_label);
    }
    break;
  case NN_MAE:
    assert( FALSE);
    
  default:
    assert( FALSE);
  }
  
  assert( FALSE);
}

nn_network nn_new( const uint32_t nbr_layers,
		   const double learning_rate,
		   const double weight_decay,
		   const uint8_t cost_func,
		   const uint8_t regularization_func,
		   const uint8_t weights_initialization,
		   const uint8_t monitor_training_cost,
		   const uint8_t monitor_training_accuracy,
		   const uint8_t monitor_evaluation_cost,
		   const uint8_t monitor_evaluation_accuracy
		   ) {

  nn_network nn = (nn_network) MEM_CALLOC( 1, sizeof( nn_network_struct));

  assert( nbr_layers > 1);
  assert( 0 < learning_rate); // && learning_rate <= 1.0);
  assert( 0 <= weight_decay && weight_decay < 1.0);
  assert( cost_func >= NN_MSE && cost_func <= NN_LOG_LIKELIHOOD);
  assert( regularization_func >= NN_NO_REGULARIZATION && regularization_func <= NN_L2_REGULARIZATION);
  
  nn->nbr_layers = nbr_layers;
  nn->learning_rate = learning_rate;
  nn->weight_decay = weight_decay;
  nn->cost_func = cost_func;
  nn->regularization_func = regularization_func;

  nn->shuffle = TRUE;
  nn->use_bias = TRUE;
  nn->weights_initialization = weights_initialization;

  nn->monitor_training_cost = monitor_training_cost;
  nn->monitor_training_accuracy = monitor_training_accuracy;
  nn->monitor_evaluation_cost = monitor_evaluation_cost;
  nn->monitor_evaluation_accuracy = monitor_evaluation_accuracy;
  
  nn->layers = (nn_layer *) MEM_CALLOC( nbr_layers, sizeof( nn_layer));

  return nn;  
}

nn_layer nn_new_layer(
		      const nn_network nn,
		      const uint32_t nbr_nodes,  // nbr of nodes in layer
		      const uint16_t act_func,   // activation function
		      const uint8_t layer_type,
		      const double dropout_rate
		      )
{
  assert( act_func >= NN_ACT_NONE && act_func <= NN_ACT_SOFTMAX);
  assert( layer_type == NN_DENSE_LAYER ||
	  ((layer_type == NN_DROPOUT_LAYER) && (dropout_rate > 0)));
  
  nn_layer l = MEM_CALLOC( 1, sizeof( nn_layer_struct));

  l->nbr_nodes = nbr_nodes;
  l->type = layer_type;

  l->act_func     = (layer_type == NN_DENSE_LAYER)?act_func:0;
  l->dropout_rate = (layer_type == NN_DROPOUT_LAYER)?dropout_rate:0.0;

  // a is output of layer
  if ( nbr_nodes == 1) {
    l->a = t_new_scalar( 0, NN_DTYPE);
  } else {
    l->a = t_new_vector( nbr_nodes, NN_DTYPE, NULL);
  }

  if ( l->type == NN_DROPOUT_LAYER) {
    assert( nbr_nodes > 1);
    l->z = NULL;
    l->dropout = t_new_vector( nbr_nodes, NN_DTYPE, NULL);    
  } else { // NN_DENSE_LAYER
    l->dropout = NULL;
    l->z = (nbr_nodes==1)?t_new_scalar( 0, NN_DTYPE):t_new_vector( nbr_nodes, NN_DTYPE, NULL);
  }
  
  return l;
}


t_tensor nn_reduce_weights( const uint8_t weight_initialization,
			    t_tensor weights,
			    const uint32_t nbr_connecting_nodes,
			    const uint32_t nbr_output_nodes) {
  // Initialize each weight using a Gaussian distribution with mean 0
  // and standard deviation 1 over the square root of the number of
  // weights connecting to the same neuron.

  assert( nbr_connecting_nodes > 0);

  double std_dev = 0;

  if ( weight_initialization == NN_WEIGHTS_NORMAL_SMALL) {
    std_dev = sqrt( (double) (1.0 / nbr_connecting_nodes));
  } else if ( weight_initialization == NN_WEIGHTS_HE_NORMAL) {
    std_dev = sqrt( (double) (2.0 / nbr_connecting_nodes));    
  } else if ( weight_initialization == NN_WEIGHTS_GLOROT_NORMAL) {
    std_dev = sqrt( (double) (2.0 / (nbr_connecting_nodes + nbr_output_nodes)));
  } else {
    assert( FALSE);
  }

  assert( std_dev > 0.0);
  t_tensor d = t_new_scalar( std_dev, NN_DTYPE);
  weights = t_multiply( weights, d, weights);
  T_FREE( d);

  return weights;
}

// initialize weights & bias for upstream layer 
static void nn_init_layer( nn_network nn, uint16_t idx, const double epsilon) {

  // invoked for all but input layer...
  assert( idx > 0);
  
  const nn_layer nl  = nn->layers[idx];     // layer l
  const nn_layer l_1 = nn->layers[idx-1];   // layer l-1: upstream

  if ( nl->type == NN_DENSE_LAYER) {
    // weights, bias, grad_b, grad_W are NULL for output layer...
    // we allocate weights & bias with the upstream layer (l-1) which can be DENSE or DROPOUT
    
    // allocate bias and weight tensors of proper dimensions: output of layer l_1 must match input of layer nl
    if ( nn->use_bias) {
      if ( nn->weights_initialization > NN_NO_INITIAL_WEIGHTS) {
	// use normally distributed random biases...
	const uint32_t shape[1] = { nl->nbr_nodes };
	l_1->bias = t_normal( 0, epsilon, 1, shape, NN_DTYPE);
      } else {
	l_1->bias = t_new_vector( nl->nbr_nodes, NN_DTYPE, NULL);
      }
      l_1->bias = t_squeeze( l_1->bias, NULL);
    } else {
      l_1->bias = NULL;
    }
    
    if ( nn->weights_initialization > NN_NO_INITIAL_WEIGHTS) {
      const uint32_t shape[2] = { nl->nbr_nodes, l_1->nbr_nodes };
      l_1->weights = t_normal( 0, epsilon, 2, shape, NN_DTYPE);
      if ( nn->weights_initialization > NN_WEIGHTS_NORMAL) {
	l_1->weights = nn_reduce_weights( nn->weights_initialization, l_1->weights, l_1->nbr_nodes, nl->nbr_nodes);
      }
    } else {
      l_1->weights = t_new_matrix( nl->nbr_nodes, l_1->nbr_nodes, NN_DTYPE, NULL);
    }
    l_1->weights = t_squeeze( l_1->weights, NULL);

    l_1->grad_b = t_new_vector( nl->nbr_nodes, NN_DTYPE, NULL);
    l_1->grad_b = t_squeeze( l_1->grad_b, NULL);
    
    l_1->grad_W = t_new_matrix( nl->nbr_nodes, l_1->nbr_nodes, NN_DTYPE, NULL);
    l_1->grad_W = t_squeeze( l_1->grad_W, NULL);

    nl->delta = t_new_vector( nl->nbr_nodes, NN_DTYPE, NULL);
    nl->delta = t_squeeze( nl->delta, NULL);

  } else { // drop-out layer

    // no successive dropout layers
    assert( l_1->type != NN_DROPOUT_LAYER);
    // nbr of nodes must match with upstream layer
    assert( l_1->nbr_nodes == nl->nbr_nodes);
    
    // no weights, bias, grad_b, grad_W for up-stream layer of DROPOUT layer
    
    // NULL on input layer
    nl->delta = t_new_vector( nl->nbr_nodes, NN_DTYPE, NULL);
    nl->delta = t_squeeze( nl->delta, NULL);

  } // layer->type
}


nn_network nn_add_layer( nn_network nn, nn_layer nl, const double epsilon) {
  uint32_t idx = 0;

  assert( nn->weights_initialization >= NN_NO_INITIAL_WEIGHTS &&
	  nn->weights_initialization <= NN_WEIGHTS_GLOROT_UNIFORM);

  // search for free layer if any
  while ( idx < nn->nbr_layers && nn->layers[idx] != NULL)
    idx++;

  if ( idx >= nn->nbr_layers) {
    fprintf( stderr, "nbr of layers exceeded: %d, no more free slots\n", nn->nbr_layers);
    return NULL;
  }

  nn->layers[idx] = nl;

  if ( idx > 0) { // fill layer n-1: weight matrix and bias vector
    nn_init_layer( nn, idx, epsilon);
  }

  if ( idx == 0) { // input layer
  } else if ( idx == nn->nbr_layers-1) { // output layer
  } else {  // hidden layer
  }

  return nn;
}

extern void dump_shape( const uint16_t rank, const t_shape shape);

static void dump_rank_shape( const char *tn,
			     const t_tensor t) {
  if ( t == NULL) {
    fprintf( stderr, "%s: NULL\n", tn);
    return;
  }

  if ( t->rank == 0) {
    fprintf( stderr, "%s: rank: %d\n", tn, t->rank);
    return;
  }
  
  fprintf( stderr, "%s: rank: %d, shape: ", tn, t->rank);
  dump_shape( t->rank, t->shape);
  fprintf( stderr, "\n");
}

// dumping weights and activations
void nn_dump_layer( const nn_layer l, const uint8_t verbose) {

  if ( l->type == NN_DENSE_LAYER) {
    if ( verbose) {
      fprintf( stderr, "z: ");
      t_dump( l->z, FALSE);
    } else {
      dump_rank_shape( "z", l->z);
    }
  } else {
    if ( verbose) {
      fprintf( stderr, "dropout: ");
      t_dump( l->dropout, FALSE);
    } else {
      dump_rank_shape( "dropout", l->dropout);
    }
  }

  if ( verbose) {
    fprintf( stderr, "a: ");
    t_dump( l->a, FALSE);
  } else {
    dump_rank_shape( "a", l->dropout);
  }

  if ( l->weights == NULL) {
    fprintf( stderr, "W: NULL\n");
  } else {
    if ( verbose) {
      fprintf( stderr, "W: \n");
      t_dump( l->weights, FALSE);
    } else {
      dump_rank_shape( "W", l->weights);
    }
  }

  if ( l->bias == NULL) {
    fprintf( stderr, "b: NULL\n");
  } else {
    if ( verbose) {
      fprintf( stderr, "b: \n");
      t_dump( l->bias, FALSE);
    } else {
      dump_rank_shape( "b", l->bias);
    }
  }
    
  fprintf( stderr, "\n");
}

char *nn_activation_to_str( const uint32_t a) {
  switch ( a) {
  case NN_ACT_NONE:
    return "none";
  case NN_ACT_SIGMOID:
    return "sigmoid";
  case NN_ACT_TANH:
    return "tanh";
  case NN_ACT_RELU:
    return "relu";
  case NN_ACT_SOFTMAX:
    return "softmax";
  default:
    return "unknown";
  }
}

char *nn_cost_func_to_str( const uint16_t f) {
  switch( f) {
  case NN_MSE: return "mean square error";
  case NN_MAE: return "mean absolute error";
  case NN_CROSS_ENTROPY: return "cross entropy";
  case NN_LOG_LIKELIHOOD: return "log likelihood";
  default:
    return "unknown";
  }
}

char *nn_reg_func_to_str( const uint16_t f) {
  switch( f) {
  case NN_NO_REGULARIZATION: return "none";
  case NN_L1_REGULARIZATION: return "L1 regularization";
  case NN_L2_REGULARIZATION: return "L2 regularization";
  default:
    return "unknown";
  }
}


void nn_dump( const nn_network nn, const uint8_t verbose) {
  
  fprintf( stderr, "depth: %d, learning rate:  %f, weight_decay: %f\n",
	   nn->nbr_layers, nn->learning_rate, nn->weight_decay);
  fprintf( stderr, "cost func: %s, ", nn_cost_func_to_str( nn->cost_func));
  fprintf( stderr, "regularization func: %s \n", nn_reg_func_to_str( nn->regularization_func));
  
  for ( int i = 0; i < nn->nbr_layers; i++) {
    const nn_layer l = nn->layers[i];

    if ( l->type == NN_DENSE_LAYER) {
      fprintf( stderr, "layer[ %d]: DENSE, #nodes = %d, activation = %s\n", i, l->nbr_nodes, nn_activation_to_str( l->act_func));
    } else {
      fprintf( stderr, "layer[ %d]: DROPOUT, #nodes = %d, dropout-rate = %f\n", i, l->nbr_nodes, l->dropout_rate);
    }
    
    if ( verbose) {
      nn_dump_layer( l, FALSE);
    }
  }
}

void nn_start_batch( const nn_network nn) {

  nn->sample_cnt = 0;

  for ( uint32_t l = 0; l < nn->nbr_layers; l++) {
    const nn_layer nl = nn->layers[l];

    // output layer
    if ( l < nn->nbr_layers-1) {
      t_clear( nl->grad_W);
      t_clear( nl->grad_b);
    }

    // not input layer
    if ( l > 0) {
      t_clear( nl->delta);
    }
  }
}


static void setup_dropout( t_tensor t, const double rate) {

  assert( t->dtype == T_FLOAT || t->dtype == T_DOUBLE);
  
  for ( uint32_t i = 0; i < t->size; i++) {

    // double r = 1;

    double r = ((double) rand()) / ((double) RAND_MAX);
    assert( r >= 0.0 && r <= 1.0);

    if ( r < rate) {
      r = 0.0;
    } else {
      // note: 1-rate is the right divisor. if rate = 0% -> r == 1
      // which means we have the identity for dropout
      r = 1.0/(1-rate);
    }

    
    switch ( t->dtype) {
    case T_FLOAT:
      {
	float *f_ptr = (float *) t->data;
	f_ptr[i] = (float) r;
      }
      break;
    case T_DOUBLE:
      {
	double *d_ptr = (double *) t->data;
	d_ptr[i] = (double) r;
      }
      break;
    default:
      assert( FALSE);
    }
  }

}

t_tensor nn_forward_dense( const nn_layer l,          // layer l
			   const nn_layer l_1,        // layer l-1, up-stream
			   const uint8_t trace, 
			   const uint8_t use_bias,
			   const uint8_t training) {
  
  if ( l->type == NN_DENSE_LAYER) {

    if ( trace) {
      fprintf( stderr, "l_1->weights: \n"); t_dump_head( l_1->weights, 10);
      fprintf( stderr, "l_1->a: \n"); t_dump_head( l_1->a, 10);
      fprintf( stderr, "l_1->bias: \n"); t_dump_head( l_1->bias, 10);
    }

    assert( t_assert( l_1->weights));
    assert( t_assert( l_1->bias));
    assert( t_assert( l_1->a));    
    
    // z = W_l_1 dot a_l_1 + bias_l_1: weighted input to hidden layer plus bias
    l->z = t_dot( l_1->weights, l_1->a, l->z);
    assert( t_assert( l->z));

    if ( use_bias) {
      if ( trace) {
	fprintf( stderr, "l->z: \n"); t_dump( l->z, FALSE);
      }
      l->z = t_add( l->z, l_1->bias, l->z);
    }
    
    if ( trace) {
      fprintf( stderr, "l->z: \n"); t_dump_head( l->z, 10);
    }

    switch ( l->act_func) {
    case NN_ACT_NONE:
      assert( FALSE); // only input layer has no activation function
    case NN_ACT_SIGMOID:
      l->a = t_sigmoid( l->z, l->a); // t_apply( l->z, sigmoid, l->a);
      break;
    case NN_ACT_TANH:
      l->a = t_tanh( l->z, l->a); // t_apply( l->z, tanh, l->a);
      break;
    case NN_ACT_RELU:
      l->a = t_relu( l->z, l->a); // t_apply( l->z, relu, l->a);
      break;
    case NN_ACT_SOFTMAX:
      l->a = softmax( l->z, l->a);
      break;
    default:
      assert( FALSE);
    }

    if ( trace) {
      fprintf( stderr, "l->a: \n"); t_dump_head( l->a, 10);
    }

  } else {

    if ( training) {
      // input to drop-out layer is l_1->a...
      // we use the z tensor as drop-out filter
      assert( t_same_shape( l->dropout, l_1->a));
      assert( t_same_shape( l->dropout, l->a));
    
      setup_dropout( l->dropout, l->dropout_rate);
      // t_dump( l->dropout, TRUE);
      
      // element-wise multiply
      l->a = t_multiply( l_1->a, l->dropout, l->a);
      // t_dump( l->a, TRUE);
      
    } else {  // evaluating
      assert( t_same_shape( l->a, l_1->a));
      t_copy( l->a, l_1->a);
    }
  }
  
}

t_tensor nn_forward( const nn_network nn,
		     const t_tensor input,
		     const uint8_t training,
		     const uint8_t trace) {

  assert( t_is1D( input));
  
  const nn_layer output_layer = nn->layers[nn->nbr_layers-1];

  const uint32_t nbr_output_nodes = output_layer->nbr_nodes;

  nn_layer l = nn->layers[0];
  assert( t_1D_len( input) == l->nbr_nodes);

  // copy input to input layer's activation nodes. 
  t_copy( l->a, input);

  // feed forward over layers, starting after input layer and including output layer
  for ( int i = 1; i < nn->nbr_layers; i++) {
    l = nn->layers[i];
    const nn_layer l_1 = nn->layers[i-1];

    nn_forward_dense( l, l_1, trace, nn->use_bias, training);

#if 0    

    if ( trace) {
      fprintf( stderr, "l_1->weights: \n"); t_dump_head( l_1->weights, 10);
      fprintf( stderr, "l_1->a: \n"); t_dump_head( l_1->a, 10);
      fprintf( stderr, "l_1->bias: \n"); t_dump_head( l_1->bias, 10);
    }
    
    // z = W_l_1 dot a_l_1 + bias_l_1: weighted input to hidden layer plus bias
    l->z = t_dot( l_1->weights, l_1->a, l->z);

    if ( nn->use_bias) {
      if ( trace) {
	fprintf( stderr, "l->z: \n"); t_dump( l->z, FALSE);
      }
      l->z = t_add( l->z, l_1->bias, l->z);
    }
    
    if ( trace) {
      fprintf( stderr, "l->z: \n"); t_dump_head( l->z, 10);
    }

    switch ( l->act_func) {
    case NN_ACT_NONE:
      assert( FALSE); // only input layer has no activation function
    case NN_ACT_SIGMOID:
      l->a = t_sigmoid( l->z, l->a); // t_apply( l->z, sigmoid, l->a);
      break;
    case NN_ACT_TANH:
      l->a = t_tanh( l->z, l->a); // t_apply( l->z, tanh, l->a);
      break;
    case NN_ACT_RELU:
      l->a = t_relu( l->z, l->a); // t_apply( l->z, relu, l->a);
      break;
    case NN_ACT_SOFTMAX:
      l->a = softmax( l->z, l->a);
      break;
    default:
      assert( FALSE);
    }

    if ( trace) {
      fprintf( stderr, "l->a: \n"); t_dump_head( l->a, 10);
    }

#endif
    
  }

  return output_layer->a; // output of last layer
}


void nn_layer_end_batch( const uint8_t regularization_func,
			 const double weight_decay,
			 const t_tensor alpha_over_m,
			 const t_tensor alpha_times_lambda,
			 const uint8_t use_bias,
			 t_tensor grad_W,
			 t_tensor weights,
			 t_tensor grad_b,
			 t_tensor bias
			 ) {

  if ( regularization_func == NN_NO_REGULARIZATION || weight_decay == 0) {

    // grad_W = (alpha / m) * grad_W
    grad_W = t_multiply( grad_W, alpha_over_m, grad_W);
    // W = W - grad_W
    weights = t_subtract( weights, grad_W, weights);

  } else if ( regularization_func == NN_L2_REGULARIZATION) {
    assert( weight_decay > 0.0);
    assert( alpha_times_lambda != NULL);

    // Nielsen (93)

    // grad_W = (alpha / m) * grad_W
    grad_W = t_multiply( grad_W, alpha_over_m, grad_W);

    // W = W * (1 - (alpha * lambda) / n)
    weights = t_multiply( weights, alpha_times_lambda, weights);

    // W = W - grad_W
    weights = t_subtract( weights, grad_W, weights);
  } else if ( regularization_func == NN_L1_REGULARIZATION) {
    assert( weight_decay > 0.0);
    assert( alpha_times_lambda != NULL);

    // Nielsen (97)

    // grad_W = (alpha / m) * grad_W
    grad_W = t_multiply( grad_W, alpha_over_m, grad_W);
      
    t_tensor sgn_w = t_sign( weights, NULL);
    // W = W - sgn(W) * ( alpha * lambda) / n
    sgn_w = t_multiply( sgn_w, alpha_times_lambda, sgn_w);
    weights = t_subtract( weights, sgn_w, weights);

    // W = W - grad_W
    weights = t_subtract( weights, grad_W, weights);

    T_FREE( sgn_w);  // w changes shape over layers...

  }

  if ( use_bias) {
    grad_b = t_multiply( grad_b, alpha_over_m, grad_b);
    bias = t_subtract( bias, grad_b, bias);
  }
 
}

t_tensor nn_get_alpha_over_m( double learning_rate, uint32_t sample_cnt) {
  double d = (double) (learning_rate/sample_cnt);
  // fprintf( stderr, "alpha_over_m: %f\n", d);
  return t_new_scalar( d, NN_DTYPE);
}

t_tensor nn_get_alpha_times_lambda( const uint8_t regularization_func,
				    const double learning_rate,
				    const double weight_decay,
				    const uint32_t nbr_samples) {
  // Nielsen (93) (97)
  double d = 0.0;
  
  if ( regularization_func == NN_L2_REGULARIZATION && weight_decay > 0) {
    // 1 - ( alpha * lambda) / n; Nielsen (93)
    d = (double) (1.0 - ((learning_rate * weight_decay) / nbr_samples));
  } else if ( regularization_func == NN_L1_REGULARIZATION && weight_decay > 0) {
    // alpha * lambda / n; Nielsen (97)
    d = (double) ((learning_rate * weight_decay) / nbr_samples);
  } else { // else no regularization or weight_decay == 0
    d = 0;
  }

  // fprintf( stderr, "alpha_times_lambda: %f\n", d);
  const t_tensor alpha_times_lambda = t_new_scalar( d, NN_DTYPE);
  return alpha_times_lambda; 
}

// nbr_samples: total nbr of samples in training set
// nn->sample_cnt: sample count over mini-batch
void nn_end_batch( const nn_network nn, const uint32_t nbr_samples) {

  assert( nn->sample_cnt > 0);
  assert( nbr_samples > 0);
  
  t_tensor alpha_over_m = nn_get_alpha_over_m( nn->learning_rate, nn->sample_cnt);
  t_tensor alpha_times_lambda = nn_get_alpha_times_lambda( nn->regularization_func,
							   nn->learning_rate,
							   nn->weight_decay,
							   nbr_samples);

  assert( alpha_over_m != NULL);
  assert( alpha_times_lambda != NULL);

#if 0
  // Nielsen (93) (97)
  t_tensor alpha_times_lambda  = NULL;

  if ( nn->regularization_func == NN_L2_REGULARIZATION && nn->weight_decay > 0) {
    // 1 - ( alpha * lambda) / n; Nielsen (93)
    alpha_times_lambda =
      t_new_scalar( (double) (1.0 - ((nn->learning_rate * nn->weight_decay) / nbr_samples)), NN_DTYPE);
  } else if ( nn->regularization_func == NN_L1_REGULARIZATION && nn->weight_decay > 0) {
    // alpha * lambda / n; Nielsen (97)
    alpha_times_lambda =
      t_new_scalar( (double) ((nn->learning_rate * nn->weight_decay) / nbr_samples), NN_DTYPE);
  } // else no regularization or weight_decay == 0
#endif
  
  // iterat over all layers, except output layer...
  for ( uint32_t l = 0; l < nn->nbr_layers-1; l++) {
    const nn_layer nl = nn->layers[l];
    const nn_layer nl_1 = nn->layers[l+1]; // down-stream

    if ( nl_1->type == NN_DROPOUT_LAYER) {
      assert( nl->type == NN_DENSE_LAYER);
      assert( nl->weights == NULL);
      continue;
    }
    
    nn_layer_end_batch( nn->regularization_func, nn->weight_decay,
			alpha_over_m, alpha_times_lambda, nn->use_bias,
			nl->grad_W, nl->weights, nl->grad_b, nl->bias);


#if 0    
    if ( nn->regularization_func == NN_NO_REGULARIZATION || nn->weight_decay == 0) {

      // grad_W = (alpha / m) * grad_W
      nl->grad_W = t_multiply( nl->grad_W, alpha_over_m, nl->grad_W);
      // W = W - grad_W
      nl->weights = t_subtract( nl->weights, nl->grad_W, nl->weights);

    } else if ( nn->regularization_func == NN_L2_REGULARIZATION) {
      assert( nn->weight_decay > 0.0);
      assert( alpha_times_lambda != NULL);

      // Nielsen (93)

      // grad_W = (alpha / m) * grad_W
      nl->grad_W = t_multiply( nl->grad_W, alpha_over_m, nl->grad_W);

      // W = W * (1 - (alpha * lambda) / n)
      nl->weights = t_multiply( nl->weights, alpha_times_lambda, nl->weights);

      // W = W - grad_W
      nl->weights = t_subtract( nl->weights, nl->grad_W, nl->weights);
    } else if ( nn->regularization_func == NN_L1_REGULARIZATION) {
      assert( nn->weight_decay > 0.0);
      assert( alpha_times_lambda != NULL);

      // Nielsen (97)

      // grad_W = (alpha / m) * grad_W
      nl->grad_W = t_multiply( nl->grad_W, alpha_over_m, nl->grad_W);
      
      t_tensor sgn_w = t_sign( nl->weights, NULL);
      // W = W - sgn(W) * ( alpha * lambda) / n
      sgn_w = t_multiply( sgn_w, alpha_times_lambda, sgn_w);
      nl->weights = t_subtract( nl->weights, sgn_w, nl->weights);

      // W = W - grad_W
      nl->weights = t_subtract( nl->weights, nl->grad_W, nl->weights);

      T_FREE( sgn_w);  // w changes shape over layers...

    }

    if ( nn->use_bias) {
      nl->grad_b = t_multiply( nl->grad_b, alpha_over_m, nl->grad_b);
      nl->bias = t_subtract( nl->bias, nl->grad_b, nl->bias);
    }

#endif
    
  }

  T_FREE( alpha_over_m);
  T_FREE( alpha_times_lambda);

}

static t_tensor derivate_activation( const nn_layer nl) {

  t_tensor t = NULL;
  
  switch( nl->act_func) {
  case NN_ACT_NONE:
    // linear activation, derivative == 1
    {
      t = t_new_tensor( nl->z->rank, nl->z->shape, nl->z->dtype, NULL);
      t_fill( t, 1.0);
      return t;
    }
  case NN_ACT_SIGMOID:
    t = t_apply( nl->z, sigmoid_derivate, NULL);
    break;
  case NN_ACT_TANH:
    t = t_apply( nl->z, tanh_derivate, NULL);
    break;
  case NN_ACT_RELU:
    t = t_apply( nl->z, relu_derivate, NULL);
    break;
  case NN_ACT_SOFTMAX:
    fprintf( stderr, "derivate_activation: softmax()\n");
    return NULL;
  default:
    fprintf( stderr, "derivate_activation: unknown activation function\n");
    return NULL;
  }
  return t;
}

static t_tensor grad_vector_abs_error( const t_tensor a, const t_tensor t) {
  assert(( t_is0D( t) && t_is0D( a)) ||
	 ( t_is1D( t) && t_is1D( a) && t_1D_len( t) == t_1D_len( a)));

  // grad( abs_err( a-t)) == sign( a-t)
  t_tensor a_minus_t = t_subtract( a, t, NULL);
  t_tensor sign_a_minus_t = t_sign( a_minus_t, NULL);

  T_FREE( a_minus_t);

  // tensor of +1, -1
  return sign_a_minus_t;
}

static t_tensor gradient_cost( const uint8_t cost_func, const t_tensor a, const t_tensor t) {

  switch( cost_func) {
  case NN_MSE:
    return t_subtract( a, t, NULL);  // a[i] - t[i]
  case NN_MAE:
    return grad_vector_abs_error( a, t); // sign( a[i] - t[i])
  default:
    fprintf( stderr, "gradient_cost: unimplemented case\n");
    return NULL;
  }
}

void nn_delta_output_layer( const nn_layer nl,
			    const t_tensor t,
			    const uint8_t cost_func) {
  // start with output layer, handling some special cases
  if ( cost_func == NN_CROSS_ENTROPY) {
    // eq ( 66), Nielsen
    assert( nl->act_func == NN_ACT_SIGMOID);
    nl->delta = t_subtract( nl->a, t, nl->delta);
  } else if ( cost_func == NN_LOG_LIKELIHOOD) {
    // eq ( 84), Nielsen
    assert( nl->act_func == NN_ACT_SOFTMAX);
    nl->delta = t_subtract( nl->a, t, nl->delta);
  } else {
    // eq ( BP1a), Nielsen
    t_tensor grad_cost = gradient_cost( cost_func, nl->a, t);
    assert( t_assert( grad_cost));

    t_tensor deriv_activation = derivate_activation( nl);
    assert( t_assert( deriv_activation));

    nl->delta = t_multiply( grad_cost, deriv_activation, nl->delta);

    T_FREE( deriv_activation);
    T_FREE( grad_cost);
  }

}

void nn_compute_delta( const nn_layer nl,
		       const nn_layer nl_1, // down-stream layer, l+1
		       const uint8_t trace,
		       const int32_t layer
		       ) {

  if ( nl->type == NN_DENSE_LAYER) {
    if ( nl_1->type == NN_DENSE_LAYER) {

      t_tensor W_nl_T = t_transpose( nl->weights, NULL);
      t_tensor W_nl_T_dot_delta_nl_1 = t_dot( W_nl_T, nl_1->delta, NULL);

      assert( t_is1D( W_nl_T_dot_delta_nl_1));

      t_tensor deriv_activation = derivate_activation( nl);

      assert( t_same_shape( deriv_activation, W_nl_T_dot_delta_nl_1));
	    
      nl->delta = t_multiply( W_nl_T_dot_delta_nl_1, deriv_activation, nl->delta);

      if ( trace) {
	fprintf( stderr, "delta(%d):\n", layer); t_dump( nl->delta, FALSE);
      }

      T_FREE( W_nl_T);
      T_FREE( W_nl_T_dot_delta_nl_1);
      T_FREE( deriv_activation);
    } else {
      // current layer is dense, downstream is drop-out
      assert( nl_1->type == NN_DROPOUT_LAYER);
      
      // dense layers up-stream of dropout layer's don't have weights.
      // which would be all 1 (identity) since no change in shape of outputs
      // thus W_nl_T == I && W_nl_T_dot_delta_nl_1 == nl_1->delta...
      assert( nl->weights == NULL);
      assert( nl->nbr_nodes == nl_1->nbr_nodes);

      // the errors depend however on the derivations of current layer
      t_tensor deriv_activation = derivate_activation( nl);
      
      nl->delta = t_multiply( nl_1->delta, deriv_activation, nl->delta);
      
      assert( t_same_shape( nl->delta, nl_1->delta));
      assert( t_same_shape( nl_1->dropout, nl_1->delta));
      
      nl->delta = t_multiply( nl->delta, nl_1->dropout, nl->delta);

      T_FREE( deriv_activation);
    }
  } else {
    // nl is DROPOUT_LAYER

    // downstream is DENSE and is assumed to have weights
    assert( nl_1->type == NN_DENSE_LAYER);
    assert( nl->weights != NULL);
    
    t_tensor W_nl_T = t_transpose( nl->weights, NULL);
    t_tensor W_nl_T_dot_delta_nl_1 = t_dot( W_nl_T, nl_1->delta, NULL);

    // no activation, y = x => derivative is 1 and we can skip the multiplication
    // with identity and have nl->delta = W_nl_T_dot_delta_nl_1
    assert( t_same_shape( nl->delta, W_nl_T_dot_delta_nl_1));
    t_copy( nl->delta, W_nl_T_dot_delta_nl_1);

    T_FREE( W_nl_T);
    T_FREE( W_nl_T_dot_delta_nl_1);
  }
}

void nn_compute_grad( const nn_layer nl,
		      const nn_layer nl_1 // l+1, down-stream layer
		      ) {

  if ( nl_1->type == NN_DROPOUT_LAYER) {
    // dense layers before drop-out layers don't have weights....
    assert( nl->type == NN_DENSE_LAYER);
    assert( nl->weights == NULL);
    return;
  }
  
  // nabla_W_l = dot( delta_l_1, a_l_T): the python dot product can become an outer product.
  // in case of (n x 1) and ( 1 x m) tensors, the dot product becomes a matrix multiplication
  // i.e. we get (n x m) matrix as result. this is the same as doing outer() product of a
  // (n x 1) and ( m x 1) vector due to the transpose used in the dot() product.

  // assert( nl_1->delta->rank == 1 || nl_1->delta->rank == 0); 
  // assert( nl->a->rank == 1);

  // in case of a singleton output node, we may have 0D delta...
  assert( t_is1D( nl_1->delta) || t_is0D( nl_1->delta));
  assert( t_is1D( nl->a));
  t_tensor nabla_W_l = t_outer( nl_1->delta, nl->a, NULL);
    
  const t_tensor nabla_b_l = nl_1->delta;  // alias

  assert( t_same_shape( nl->grad_W, nabla_W_l));
  nl->grad_W = t_add( nl->grad_W, nabla_W_l, nl->grad_W);

  assert( t_same_shape( nl->grad_b, nabla_b_l));
  nl->grad_b = t_add( nl->grad_b, nabla_b_l, nl->grad_b);

  T_FREE( nabla_W_l);
}

void nn_backprop( const nn_network nn,
		  const t_tensor t,
		  const uint8_t trace) {

  // output layer
  nn_layer nl = nn->layers[nn->nbr_layers-1];

  assert(( t_is0D( t) && t_is0D( nl->a)) ||
	 ( t_is1D( t) && t_is1D( nl->a) && t_1D_len( t) == t_1D_len( nl->a)));
  
  // compute delta_nl
  nn_delta_output_layer( nl, t, nn->cost_func);

#if 0
  // start with output layer, handling some special cases
  if ( nn->cost_func == NN_CROSS_ENTROPY) {
    // eq ( 66), Nielsen
    assert( nl->act_func == NN_ACT_SIGMOID);
    nl->delta = t_subtract( nl->a, t, nl->delta);
  } else if ( nn->cost_func == NN_LOG_LIKELIHOOD) {
    // eq ( 84), Nielsen
    assert( nl->act_func == NN_ACT_SOFTMAX);
    nl->delta = t_subtract( nl->a, t, nl->delta);
  } else {
    // eq ( BP1a), Nielsen
    t_tensor grad_cost = gradient_cost( nn->cost_func, nl->a, t);
    assert( t_assert( grad_cost));

    t_tensor deriv_activation = derivate_activation( nl);
    assert( t_assert( deriv_activation));

    nl->delta = t_multiply( grad_cost, deriv_activation, nl->delta);

    T_FREE( deriv_activation);
    T_FREE( grad_cost);
  }
#endif
  
  if ( trace) {
    fprintf( stderr, "delta(Nl):\n"); t_dump( nl->delta, FALSE);
    fprintf( stderr, "a(Nl):\n"); t_dump( nl->a, FALSE);
    fprintf( stderr, "t: \n"); t_dump( t, FALSE);
  }
  
  // go backwards from layer before output layer (index == nbr_layers-2)
  // towards input layer, excluding input layer and compute delta for all
  // non output layers and non input layers
  for ( int32_t l = nn->nbr_layers-2; l > 0; l--) {
    nl = nn->layers[l];
    nn_layer nl_1 = nn->layers[l+1]; // down-stream

    nn_compute_delta( nl, nl_1, trace, l);

    assert( t_is1D( nl->delta));

    if ( trace) {
      fprintf( stderr, "delta(%d):\n", l); t_dump( nl->delta, FALSE);
    }
    
#if 0    

    t_tensor W_nl_T = t_transpose( nl->weights, NULL);
    t_tensor W_nl_T_dot_delta_nl_1 = t_dot( W_nl_T, nl_1->delta, NULL);

    assert( t_is1D( W_nl_T_dot_delta_nl_1));

    t_tensor deriv_activation = derivate_activation( nl);

    assert( t_same_shape( deriv_activation, W_nl_T_dot_delta_nl_1));
	    
    nl->delta = t_multiply( W_nl_T_dot_delta_nl_1, deriv_activation, nl->delta);

    if ( trace) {
      fprintf( stderr, "delta(%d):\n", l); t_dump( nl->delta, FALSE);
    }

    T_FREE( W_nl_T);
    T_FREE( W_nl_T_dot_delta_nl_1);
    T_FREE( deriv_activation);

 #endif
	    
  }

  // we now have deltas for all but input layer
  // but must compute weights and biases including input layer using layer n+1 delta...
  // going backwards
  for ( int32_t l = nn->nbr_layers-2; l >= 0; l--) {
    nl = nn->layers[l];
    nn_layer nl_1 = nn->layers[l+1]; // down-stream

    nn_compute_grad( nl, nl_1);

#if 0
    
    // nabla_W_l = dot( delta_l_1, a_l_T): the python dot product can become an outer product.
    // in case of (n x 1) and ( 1 x m) tensors, the dot product becomes a matrix multiplication
    // i.e. we get (n x m) matrix as result. this is the same as doing outer() product of a
    // (n x 1) and ( m x 1) vector due to the transpose used in the dot() product.
    t_tensor nabla_W_l = t_outer( nl_1->delta, nl->a, NULL);
    
    const t_tensor nabla_b_l = nl_1->delta;  // alias

    assert( t_same_shape( nl->grad_W, nabla_W_l));
    nl->grad_W = t_add( nl->grad_W, nabla_W_l, nl->grad_W);

    assert( t_same_shape( nl->grad_b, nabla_b_l));
    nl->grad_b = t_add( nl->grad_b, nabla_b_l, nl->grad_b);

    T_FREE( nabla_W_l);

#endif
    
  }

  if ( trace) {
    // trace in order of layers...
    for ( uint32_t i = 0; i < nn->nbr_layers-1; i++) {
      nl = nn->layers[i];
      fprintf( stderr, "grad_W(%d): \n", i); t_dump( nl->grad_W, 0);
      fprintf( stderr, "grad_b(%d): \n", i); t_dump( nl->grad_b, 0);
    }
  }
  
}


static t_tensor get_shuffle_idx( const t_tensor t) {
  uint8_t dtype;

  const uint32_t len = t->shape[0];
  if ( len < CHAR_MAX) {
    dtype = T_INT8;
  } else if ( len < SHRT_MAX) {
    dtype = T_INT16;
  } else if ( len < INT_MAX) {
    dtype = T_INT32;
  } else {
    dtype = T_INT64;
  }

  const uint32_t shape[1] = { len };
  const t_tensor s_idx = t_new( 1, shape, dtype);

  for ( uint32_t i = 0; i < len; i++) {
    t_value idx;
    idx.dtype = T_DOUBLE;
    // draw random nbr [0..len]
    idx.u.d = (((double) rand()) / ((double) RAND_MAX)) * ((double) len);
    assign_to_tensor( s_idx, i, idx, TRUE);
  }

  return s_idx;
}

				
void nn_shuffle( t_tensor x_data, t_tensor t_data) {
  assert( t_assert( x_data));
  assert( t_assert( t_data));

  assert( t_data->shape[0] == x_data->shape[0]);

  // need to shuffle both tensors the same way....
  t_tensor s_idx = get_shuffle_idx( x_data);

  const uint32_t len = t_1D_len( s_idx);

  // buffer to exchange rows
  uint8_t *bufr_x = NULL;
  uint8_t *bufr_t = NULL;
  
  for ( uint32_t i = 0; i < len; i++) {

    t_value tgt;
    assign_to_value( &tgt, s_idx, i, T_INT64);

    t_exchange_lowest_dim( x_data, i, tgt.u.ll, &bufr_x);
    t_exchange_lowest_dim( t_data, i, tgt.u.ll, &bufr_t);
    
  }

  MEM_FREE( bufr_x); MEM_FREE( bufr_t);
  
  T_FREE( s_idx);
  
}

t_tensor nn_arange( const uint64_t lo, const uint64_t hi, const uint64_t step) {
  assert( lo < hi && step > 0);
  uint64_t cnt = (uint64_t) ( ceil( (hi - lo)/step));
  
  t_tensor t = t_new_vector( cnt, T_INT64, NULL);
  
  t_value tv;
  tv.dtype = T_INT64;
  tv.u.ll = lo;
  
  for ( uint64_t i = 0; i < cnt; i++) {
    assign_to_tensor( t, i, tv, FALSE);
    tv.u.ll += step;
  }
  return t;
}

static int64_t get_idx( const uint32_t i, const t_tensor idx) {
  t_value tv;
  assign_to_value( &tv, idx, i, T_INT64);
  return tv.u.ll;
}

// the training truth data is expected to be one-hot vectors in some cases...
static uint8_t check_training_truth_data( const nn_network nn,
					  const t_tensor t) {

  const nn_layer output_layer = nn->layers[nn->nbr_layers-1];

  assert( t_assert( t));

  // check that width of truth tensor matches nbr of output layer nodes
  if ( output_layer->nbr_nodes == 1) {
    if ( !t_is1D( t)) {
      fprintf( stderr, "expected 1D truth values\n");
      return FALSE;
    }
  } else {
    if ( !t_is2D( t)) {
      fprintf( stderr, "expected 2D truth values\n");      
      return FALSE;
    }
  }
  
  if ( output_layer->act_func == NN_ACT_TANH) {
    // truth values must be [-1..+1]
    if ( !t_check_values( t, -1.0, +1.0)) {
      fprintf( stderr, "truth tensor values not [-1.0..+1.0]\n");
      return FALSE;
    }
  } else if ( output_layer->act_func == NN_ACT_SIGMOID ||
	      output_layer->act_func == NN_ACT_SOFTMAX) {
    // truth values must be [0..+1]
    if ( !t_check_values( t, 0.0, +1.0)) {
      fprintf( stderr, "truth tensor values not [0.0..+1.0]\n");
      return FALSE;
    }
  }
  return TRUE;
}


void nn_train( const nn_network nn,     // neural network
	       const t_tensor x_data,   // training data, normalized
	       const t_tensor t_data,   // training truth data, vectorized as one-hot vector if shape[1] > 1, else not
	       const uint32_t batch_size,  // mini match size for averaging gradient
	       const uint32_t nbr_epochs,  // how often do we iterate over testing data
	       const t_tensor test_x_data, // testing data, normalized
	       const t_tensor test_t_data, // testing truth data, not vectorized
	       const uint8_t trace
	       ) {

  t_tensor input = NULL;
  t_tensor truth = NULL;

  assert( t_assert( x_data));
  assert( t_assert( t_data));

  assert( x_data->rank >= 1);
  assert( t_data->rank >= 1);
  
  const uint32_t nbr_samples = x_data->shape[0];

  fprintf( stderr, "nbr_samples: %d, batch_size: %d, nbr_epochs: %d\n",
	   nbr_samples, batch_size, nbr_epochs);

  assert( nbr_samples == t_data->shape[0]);
  assert( batch_size <= nbr_samples);

  const uint32_t index_len = 1;
  uint32_t index[index_len];

  // for evaluation we need to know if the truth data has been vectorized or not...
  const uint8_t truth_vectorized = t_data->shape[1] > 1;
  
  if ( !check_training_truth_data( nn, t_data)) {
    fprintf( stderr, "nn_train: training truth data is not in format matching output activation function\n");
    return;
  }

  if ( !t_check_values( x_data, 0.0, 1.0)) {
    fprintf( stderr, "nn_train: training data is not normalized?\n");
    return;
  }

  if ( !t_check_values( test_x_data, 0.0, 1.0)) {
    fprintf( stderr, "nn_train: testing data is not normalized?\n");
    return;
  }

  for ( uint32_t i = 0; i < nbr_epochs; i++) {

    fprintf( stderr, "epoch [%d]: \n", i);
    
    t_tensor s_idx = NULL;
    
    // shuffle input data tensors. we do this by getting a shuffled index
    // which we use for both tensors, samples and truths
    if ( nn->shuffle) {
      s_idx = t_shuffled_index( x_data);
    } else {
      // sequential index ...
      s_idx = nn_arange( 0, x_data->shape[0], 1);
    }

    uint64_t sample_cnt = 0;
    
    // iterate over all of data samples in batch-size chunks...
    for ( uint32_t j = 0; j < nbr_samples; j += batch_size) {
      
      nn_start_batch( nn);

      // iterate over mini batch
      for ( uint32_t m = 0;
	    (m < batch_size);
	    m++) {

	if ( m + j >= nbr_samples) {
	  if ((( m + j) % ( LOG_DOTS_PER_LINE * 1000)) != 0) {
	    fprintf( stderr, "\n");
	  }
	  break;
	}
	
	nn_log_progress( "sample", sample_cnt++, 1000);      

	// get index of shuffle to extract one sample & truth
	// note we get one sample out of current mini-batch...
	index[0] = (uint32_t) get_idx( m+j, s_idx);

	// here we convert data types into NN_DTYPE for one sample.
	input = t_extract( x_data, index, index_len, input, NN_DTYPE);
	truth = t_extract( t_data, index, index_len, truth, NN_DTYPE);

	if ( trace) {
	  fprintf( stderr, "sample[%d]:\n", m);
	  fprintf( stderr, "input:\n"); t_dump( input, FALSE);
	  fprintf( stderr, "truth:\n"); t_dump( truth, FALSE);
	  fprintf( stderr, "\n");
	}

	const t_tensor output = nn_forward( nn, input, /* training */ TRUE, /* trace */ FALSE);
      
	nn_backprop( nn, truth, FALSE);

	// sample count for mini-batch
	nn->sample_cnt++;

      } // for samples in mini-batch...

      nn_end_batch( nn, nbr_samples);

      if ( trace) {
	nn_dump( nn, TRUE);
      }

    } // for mini-batch...
    
    T_FREE( s_idx);

    if ( nn->monitor_training_accuracy) {
      nn_evaluate( nn, x_data, t_data, truth_vectorized, FALSE);
    }
    
    if ( nn->monitor_evaluation_accuracy) {
      nn_evaluate( nn, test_x_data, test_t_data, FALSE, FALSE);
    }

    if ( nn->monitor_training_cost) {
      nn_total_cost( nn, x_data, t_data, truth_vectorized, FALSE);
    }

    if ( nn->monitor_evaluation_cost) {
      nn_total_cost( nn, test_x_data, test_t_data, FALSE, FALSE);
    }
    
  } // for epoch iterations

  T_FREE( input);
  T_FREE( truth);
}

// assuming a being a softmax output value, return index of max value
// returns (0..len(a)], a assumed to be 1D
uint32_t nn_argmax( const t_tensor a) {
  assert( a->rank == 1);
  uint32_t index[1][a->rank];
  t_argmax( a, -1, (uint32_t *) index, 1);
  return index[0][0];
}

// determine the position of the 1 bit in vectorized truth data.
uint32_t nn_get_truth_label( const t_tensor vt, const uint32_t m) {
  assert( t_assert( vt));
  assert( m >= 0 && m < vt->shape[0]);  // valid row index

  // vectorized input is a 2D tensor of T_INT8....
  assert( t_is2D( vt));
  assert( vt->dtype == T_INT8);

  // need to get the row of vt and determine the position of the 1 bit...
  uint32_t row_idx = m * vt->strides[0] * t_dtype_size( T_INT8);

  // getting ptr to start of row
  int8_t *data = (int8_t *) vt->data;
  data += row_idx;

  // iterating over columns...
  for ( uint32_t i = 0; i < vt->shape[1]; i++) {
    const int8_t tt = data[i];
    if ( tt == 1) {
      return i;
    } else {
      // we expect values to be 0 or 1...
      assert( tt == 0);
    }
  }
  // at least one bit must be set...
  assert( FALSE);  
}

// returns nbr of nodes in output layer.
uint32_t nn_output_width( const nn_network nn) {
  const nn_layer l = nn->layers[nn->nbr_layers-1];
  return l->nbr_nodes;
}

double nn_total_cost( const nn_network nn,
		      const t_tensor x_data,
		      const t_tensor t_data,
		      const uint8_t is_vectorized,
		      const uint8_t trace) {

  // nothing to evaluate
  if ( x_data == NULL || t_data == NULL)
    return 0.0;

  assert( nn != NULL);
  
  assert( t_assert( x_data));
  assert( t_assert( t_data));

  assert( x_data->rank >= 1);
  assert( t_data->rank >= 1);

  assert(( is_vectorized && t_is2D( t_data)) ||
	 ( !is_vectorized && t_is1D( t_data)));
	 
  const uint32_t nbr_samples = x_data->shape[0];
  assert( nbr_samples == t_data->shape[0]);

  fprintf( stderr, "nn_total_cost: #samples: %d\n", nbr_samples);
  
  t_tensor input = NULL;

  const uint32_t one_hot_vector_len = nn_output_width( nn);

  double cost = 0;

  for ( uint32_t m = 0; m < nbr_samples; m++) {

    // extract test sample
    const uint32_t index[1] = {m};

    input = t_extract( x_data, index, 1, input, NN_DTYPE);

    nn_log_progress( "sample", m, 1000);

    // feedforward. the computed output is the result of the final layer's activation.
    // we get a 1D vector of output width...
    t_tensor a = nn_forward( nn, input, FALSE, FALSE);

    if ( a->rank == 0) { // singleton output
      assert( !is_vectorized);
      t_tensor truth = t_new_scalar( t_1D_get( t_data, m), T_INT8);
      
      cost += nn_cost( nn, a, truth)/nbr_samples;

      T_FREE( truth);
      
    } else {

      t_tensor truth_vect = NULL;

      // we need vectorized truth data here...
      if ( is_vectorized) {
	// in case of training data, we have already one-hot vectors.
	// just extract one row of tensor
	truth_vect = t_extract( t_data, index, 1, truth_vect, T_INT8);
      } else {
	// in case of test data, we have a 1D tensor of truth labels...
	const uint32_t tl = (uint32_t) t_1D_get( t_data, m);
	truth_vect = t_to_one_hot_vector( tl, one_hot_vector_len, truth_vect);
      }
      assert( t_assert( truth_vect));
      assert( t_is1D( truth_vect));

      cost += nn_cost( nn, a, truth_vect)/nbr_samples;

      T_FREE( truth_vect);
    }
  }

  T_FREE( input); 

  if ( nn->regularization_func != NN_NO_REGULARIZATION && nn->weight_decay != 0) {
    // handle regularization
    double w_cost = 0;

    // cost += 0.5*(lmbda/len(data))*sum( np.linalg.norm(w)**2 for w in self.weights)
    // iterate over all layers
    for ( uint32_t l = 0; l < nn->nbr_layers-1; l++) {
      const nn_layer nl = nn->layers[l];
      const nn_layer nl_1 = nn->layers[l+1]; // downstream

      if ( nl_1->type == NN_DROPOUT_LAYER) {
	assert( nl->type == NN_DENSE_LAYER);
	continue;
      }
      
      switch ( nn->regularization_func) {
      case NN_L2_REGULARIZATION:
	{
	  double norm = t_norm( nl->weights, T_ORD_FROBENIUS);
	  norm = norm * norm;
	  w_cost += norm;
	}
	break;
      case NN_L1_REGULARIZATION:
	{
	  t_tensor t = t_abs( nl->weights, NULL);
	  t_tensor s = t_sum( t, NULL);
	  assert( t_is0D( s));
	  w_cost += t_scalar( s);
	  T_FREE( t);
	  T_FREE( s);
	}
	break;
      default:
	assert( FALSE);
      } // switch
    } // over all layers

    w_cost = 0.5 * (nn->weight_decay / nbr_samples) * w_cost;

    cost += w_cost;
  }

  fprintf( stderr, "#samples: %d, cost: %f\n", nbr_samples, cost);
  
  return cost;
}


// is_vectorized: if TRUE, using vectorized input truth data which is the case for training data.
//     if FALSE, using non-vectorized input truth data, which is the case for test/evaluation data.
void nn_evaluate( const nn_network nn,
		  const t_tensor x_data,
		  const t_tensor t_data,
		  const uint8_t is_vectorized,  
		  const uint8_t trace) {

  // nothing to evaluate
  if ( x_data == NULL || t_data == NULL)
    return;

  assert( nn != NULL);
  
  assert( t_assert( x_data));
  assert( t_assert( t_data));

  assert( x_data->rank >= 1);
  assert( t_data->rank >= 1);
  
  assert(( is_vectorized && t_is2D( t_data)) ||
	 ( !is_vectorized && t_is1D( t_data)));
	 
  const uint32_t nbr_samples = x_data->shape[0];

  fprintf( stderr, "nn_evaluate: #samples: %d\n", nbr_samples);
  
  assert( nbr_samples == t_data->shape[0]);

  t_tensor input = NULL;

  uint32_t match_cnt = 0;

  for ( uint32_t m = 0; m < nbr_samples; m++) {

    // extract test sample
    const uint32_t index[1] = {m};

    input = t_extract( x_data, index, 1, input, NN_DTYPE);

    if ( trace) {
      fprintf( stderr, "sample[%d]:\n", m);
      fprintf( stderr, "input:\n"); t_dump( input, FALSE);
    } else {
      nn_log_progress( "sample", m, 1000);
    }

    // feedforward. the computed output is the result of the final layer's activation.
    t_tensor a = nn_forward( nn, input, FALSE, FALSE);

    if ( a->rank == 0) {
      double y_hat = t_scalar( a);
      uint32_t t = (uint32_t) t_1D_get( t_data, m);

      // fprintf( stderr, "t = %d, y_hat = %f\n", t, y_hat);

      if (( t == 1 && y_hat >= 0.5) || (t == 0 && y_hat < 0.5)) {
	match_cnt++;
      }
      
    } else {
      uint32_t y_hat = nn_argmax( a);

      // if we need to convert, we are using training data, i.e. vectorized truth labels
      uint32_t t = is_vectorized?nn_get_truth_label( t_data, m):((uint32_t) t_1D_get( t_data, m));

      // fprintf( stderr, "t = %d, y_hat = %d\n", t, y_hat);

      if ( y_hat == t) {
	match_cnt++;
      }
    }

  }

  T_FREE( input);

  const double perf = (double) match_cnt/nbr_samples;

  // beautiful output..
  if (( nbr_samples % ( LOG_DOTS_PER_LINE * 1000)) != 0) {
    fprintf( stderr, "\n");
  }
  
  fprintf( stderr, "#samples: %d, #matches: %d, performance: %.2f %% \n",
	   nbr_samples, match_cnt, (perf * 100.0));

}

#define F_WRITE( ptr, sz, cnt, f) { int64_t n = fwrite( ptr, sz, cnt, f); assert( n == cnt); }
#define F_READ( ptr, sz, cnt, f) { int64_t n = fread( ptr, sz, cnt, f); assert( n == cnt); }

static void write_layer( const nn_network nn, const uint32_t l, FILE *f) {
  const nn_layer nl = nn->layers[l];
  
  F_WRITE( &nl->type, sizeof( uint8_t), 1, f);
  F_WRITE( &nl->nbr_nodes, sizeof( uint32_t), 1, f);
  F_WRITE( &nl->act_func, sizeof( uint16_t), 1, f);

  const uint8_t have_weights = nl->weights != NULL;
  F_WRITE( &have_weights, sizeof( uint8_t), 1, f);

  // not all layers have weights...
  if ( have_weights) {
    assert( nl->weights != NULL && nl->bias != NULL);
    t_write( nl->weights, f);
    t_write( nl->bias, f);    
  } else {
  }
}

void nn_write( const nn_network nn, const char *fn) {
  FILE *f = fopen( fn, "wb");

  if ( f == NULL) {
    fprintf( stderr, "nn_write: failure to open %s\n", fn);
    return;
  }

  F_WRITE( &(nn->nbr_layers), sizeof( uint16_t), 1, f);
  F_WRITE( &(nn->learning_rate), sizeof( double), 1, f);
  F_WRITE( &(nn->weight_decay), sizeof( double), 1, f);
  F_WRITE( &(nn->cost_func), sizeof( uint8_t), 1, f);
  F_WRITE( &(nn->regularization_func), sizeof( uint8_t), 1, f);

  F_WRITE( &(nn->shuffle), sizeof( uint8_t), 1, f);  
  F_WRITE( &(nn->use_bias), sizeof( uint8_t), 1, f);  
  F_WRITE( &(nn->weights_initialization), sizeof( uint8_t), 1, f);

  for ( int i = 0; i < nn->nbr_layers; i++) {
    write_layer( nn, i, f);
  }
  
  fclose( f);
}

static void read_layer( const nn_network nn, const uint32_t l, FILE *f) {

  nn_layer nl = MEM_CALLOC( sizeof( nn_layer_struct), 1);
  
  F_READ( &nl->type, sizeof( uint8_t), 1, f);
  F_READ( &nl->nbr_nodes, sizeof( uint32_t), 1, f);
  F_READ( &nl->act_func, sizeof( uint16_t), 1, f);

  uint8_t have_weights = 0;
  F_READ( &have_weights, sizeof( uint8_t), 1, f);

  if ( have_weights) {
    nl->weights = t_read( f);
    nl->bias = t_read( f);    
  }

  if ( nl->type == NN_DENSE_LAYER) {
    nl->z = (nl->nbr_nodes==1)?t_new_scalar( 0, NN_DTYPE):t_new_vector( nl->nbr_nodes, NN_DTYPE, NULL);
    nl->dropout = NULL;
  } else {
    assert( nl->nbr_nodes > 0);
    nl->z = NULL;
    nl->dropout = t_new_vector( nl->nbr_nodes, NN_DTYPE, NULL);    
  }
  
  nl->a = (nl->nbr_nodes==1)?t_new_scalar(0, NN_DTYPE):t_new_vector( nl->nbr_nodes, NN_DTYPE, NULL);


  nn->layers[l] = nl;
}


nn_network nn_read( const char *fn, const uint8_t alloc_delta,
		    const uint8_t monitor_training_cost,
		    const uint8_t monitor_training_accuracy,
		    const uint8_t monitor_evaluation_cost,
		    const uint8_t monitor_evaluation_accuracy
		    ) {
  FILE *f = fopen( fn, "r");

  if ( f == NULL) {
    fprintf( stderr, "nn_read: failure to open %s\n", fn);
    return NULL;
  }

  nn_network nn = MEM_CALLOC( sizeof( nn_network_struct), 1);
  
  F_READ( &(nn->nbr_layers), sizeof( uint16_t), 1, f);
  nn->layers = MEM_CALLOC( sizeof( nn_layer), nn->nbr_layers);
  
  F_READ( &(nn->learning_rate), sizeof( double), 1, f);
  F_READ( &(nn->weight_decay), sizeof( double), 1, f);
  F_READ( &(nn->cost_func), sizeof( uint8_t), 1, f);
  F_READ( &(nn->regularization_func), sizeof( uint8_t), 1, f);
  
  F_READ( &(nn->shuffle), sizeof( uint8_t), 1, f);  
  F_READ( &(nn->use_bias), sizeof( uint8_t), 1, f);  
  F_READ( &(nn->weights_initialization), sizeof( uint8_t), 1, f);

  for ( int i = 0; i < nn->nbr_layers; i++) {
    read_layer( nn, i, f);
  }

  if ( alloc_delta) {
    for ( uint32_t i = 1; i < nn->nbr_layers; i++) {

      const nn_layer l_1 = nn->layers[i-1];  // layer l-1: upstream
      const nn_layer nl = nn->layers[i];

      l_1->grad_b = t_new_vector( nl->nbr_nodes, NN_DTYPE, NULL);
      l_1->grad_b = t_squeeze( l_1->grad_b, NULL);
    
      l_1->grad_W = t_new_matrix( nl->nbr_nodes, l_1->nbr_nodes, NN_DTYPE, NULL);
      l_1->grad_W = t_squeeze( l_1->grad_W, NULL);

      // NULL on input layer
      nl->delta = t_new_vector( nl->nbr_nodes, NN_DTYPE, NULL);
      nl->delta = t_squeeze( nl->delta, NULL);
      
    }
  } else {
    // deltas remain NULL, only needed for backprop....
  }
  
  fclose( f);

  nn->monitor_training_cost = monitor_training_cost;
  nn->monitor_training_accuracy = monitor_training_accuracy;
  nn->monitor_evaluation_cost = monitor_evaluation_cost;
  nn->monitor_evaluation_accuracy = monitor_evaluation_accuracy;

  return nn;

}
