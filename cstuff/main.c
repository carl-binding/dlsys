
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#include "logger.h"
#include "mem.h"

#include "tensor.h"
#include "mnist.h"
#include "nn_basic.h"
#include "optim.h"

#define TRUE 1
#define FALSE 0

#if 0

static void test_vars() {

  const double vv[] = { 1.1, 2.2, 3.3};
  const double mm[] = { 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9};
  const double nn[] = { 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8,
			9.9, 11.11, 12.12, 13.13, 14.14, 15.15, 16.16, 17.17, 18.18, 19.19};

  const double rr[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23};

  Value v = v_tensor( t_new_matrix( 3, 3, T_FLOAT, rr));
  Value w = v_negate( v);
  v_dump( w, 2);

  Value r = v_add( v, w);
  v_dump( r, 2);
  v_free( r);

  r = v_sub( v, w);
  v_dump( r, 2);
  v_free( r);

  r = v_mul( v, w);
  v_dump( r, 2);
  v_free( r);

  r = v_div( v, w);
  v_dump( r, 2);
  v_free( r);

  const double twos[] = { 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
  Value two = v_tensor( t_new_matrix( 3, 3, T_FLOAT, twos));
  r = v_power( v, two);
  v_dump( r, 2);
  v_free( r);
  v_free( two);

  r = v_add_scalar( v, 2.0);
  v_dump( r, 2);
  v_free( r);
  
  r = v_mul_scalar( v, 2.0);
  v_dump( r, 2);
  v_free( r);
  
  r = v_div_scalar( v, 2.0);
  v_dump( r, 2);
  v_free( r);

  {
    const double rr[] = { 0, -1, 2, -3, 4, -5, 6, -7, 8, -9, 10, -11, 12, -13, 14, -15, 16, -17, 18, -19, 20, -21, 22, -23};

    Value v = v_tensor( t_new_matrix( 3, 3, T_FLOAT, rr));
    Value w = v_sign( v);
    v_dump( w, 2);
    v_free( w);
    v_free( v);
  }
  
  r = v_power_scalar( v, 2.0);
  v_dump( r, 2);
  v_free( r);

  r = v_exp( v);
  v_dump( r, 2);
  Value s = v_log( r);
  v_dump( s, 2);
  v_free( s);

  {
    const double rr[] = { 0, -1, 2, -3, 4, -5, 6, -7, 8, -9, 10, -11, 12, -13, 14, -15, 16, -17, 18, -19, 20, -21, 22, -23};

    Value v = v_tensor( t_new_matrix( 3, 3, T_FLOAT, rr));
    Value w = v_relu( v);
    v_dump( w, 2);
    v_free( w);

    w = v_relu_deriv( v);
    v_dump( w, 2);
    v_free( w);
    v_free( v);
  }

  r = v_matmul( v, v);
  v_dump( v, 2);
  v_dump( r, 2);
  v_free( r);

  r = v_transpose( v, 0, NULL);
  v_dump( r, 2);
  v_free( r);

  {
    const double rr[]= {1,2,3,4};
    Value v = v_tensor( t_new_vector( 4, T_FLOAT, rr));
    Value w = v_transpose( v, 0, NULL);
    v_dump( v, 2);
    v_dump( w, 2);
    v_free( v);
    v_free( w);
  }

  {
    uint32_t shape[3] = { 1, 2, 3};
    Value v = v_ones( 3, shape);
    uint8_t axes[3] = { 1, 0, 2};
    
    Value w = v_transpose( v, 3, axes);
    v_dump( v, 2);
    v_dump( w, 2);
    v_free( v);
    v_free( w);
  }

  {
    uint32_t shape[4] = { 2, 3, 4, 5};
    Value v = v_ones( 4, shape);
    
    Value w = v_transpose( v, 0, NULL);
    v_dump( v, 2);
    v_dump( w, 2);
    v_free( v);
    v_free( w);
  }

  {
    uint32_t shape[2] = {5, 1};
    uint32_t n_shape[2] = {5, 6};

    Value s = v_tensor( t_new_scalar( 1.0, T_FLOAT));
    Value r = v_broadcast( s, 2, shape);
    v_dump( s, 2);
    v_dump( r, 2);
    v_free( r);

    r = v_broadcast( s, 2, n_shape);
    v_dump( r, 2);
    v_free( r);
    v_free( s);

    shape[0] = 1; shape[1] = 6;
    s = v_tensor( t_ones( 2, shape, T_FLOAT));
    r = v_broadcast( s, 2, n_shape);
    v_dump( r, 2);
    v_free( r);
    v_free( s);

    shape[0] = 6; shape[1] = 0;
    s = v_tensor( t_ones( 1, shape, T_FLOAT));
    r = v_broadcast( s, 2, n_shape);
    v_dump( r, 2);
    v_free( r);
    v_free( s);

    s = v_tensor( t_new_scalar( 2.0, T_FLOAT));
    r = v_broadcast( s, 2, n_shape);
    v_dump( r, 2);
    v_free( r);
    v_free( s);
    
  }

  {
    uint32_t shape[2] = {2, 3};
    double d[6] = { 1, 2, 3, 4, 5, 6};

    uint32_t n_shape[2] = {6, 0};

    // (2, 3) -> (6)
    Value s = v_tensor( t_new_matrix( 2, 3, T_FLOAT, d));
    Value r = v_reshape( s, 1, n_shape);
    v_dump( r, 2);
    v_free( r);

    // (2, 3) -> (3, 2)
    n_shape[0] = 3; n_shape[1] = 2;
    r = v_reshape( s, 2, n_shape);
    v_dump( r, 2);
    v_free( r);

  }
  
  {
    double d[2] = {0.5, 1.5};
    Value s = v_tensor( t_new_vector( 2, T_FLOAT, d));
    Value r = v_summation( s, 0, NULL, FALSE);
    v_dump( r, 2);
    v_free( r);

    r = v_summation( s, 0, NULL, TRUE);
    v_dump( r, 2);
    v_free( r);
    
  }

  {
    double d[4] = {0, 1, 0, 5};
    Value s = v_tensor( t_new_matrix( 2, 2, T_FLOAT, d));
    Value r = v_summation( s, 0, NULL, FALSE);
    v_dump( r, 2);
    v_free( r);

    uint8_t axes[2] = {1, 0};
    r = v_summation( s, 2, axes, FALSE);
    v_dump( r, 2);
    v_free( r);
    
    axes[0] = 0; axes[1] = 1;
    r = v_summation( s, 2, axes, FALSE);
    v_dump( r, 2);
    v_free( r);
    
    r = v_summation( s, 2, axes, TRUE);
    v_dump( r, 2);
    v_free( r);

    v_free( s);
  }


  {
    const double rr[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23};
    Value s = v_tensor( t_new_matrix( 4, 4, T_FLOAT, rr));
    Value lsm = v_log_soft_max( s);
    v_dump( s, 2);
    v_dump( lsm, 2);
    Value e = v_exp( lsm);
    v_dump( e, 2);

    uint8_t axes[2] = {0,1};
    uint32_t shape[2] = {4, 4};
    Value ones = v_ones( 2, shape);
    // expect log( 4 * e^1) == log (4e) == 2.386
    Value lse = v_log_sum_exp( ones, 2, axes);
    v_dump( lse, 2);

    v_free( ones);
    v_free( lse);
    v_free( lsm);
    v_free( e);
    
  }

  exit( 0);
  
}
#endif

static mdl_module create_residual_block( const uint32_t dim,
					 const uint32_t hidden_dim,
					 const uint8_t norm_type,
					 float drop_prob,
					 float norm_eps,
					 float norm_momentum) {

  assert( norm_type == MDL_BATCH_NORM_1D || norm_type == MDL_LAYER_NORM_1D);
  assert( dim > 0);
  assert( hidden_dim > 0);
  assert( norm_eps > 0.0);
  assert( norm_momentum > 0.0);
  
  l_list modules = l_new( 10, T_PTR, NULL);
  mdl_module lin = (mdl_module) mdl_linear_new( dim, hidden_dim);
  assert( mdl_is_module( lin));
  l_append_ptr( modules, lin);

  mdl_module norm = NULL;
  if ( norm_type == MDL_BATCH_NORM_1D) {
    norm = (mdl_module) mdl_batch_norm1D_new( hidden_dim,
					      norm_eps,
					      norm_momentum);
  } else {
    assert( FALSE);
  }
  l_append_ptr( modules, norm);

  l_append_ptr( modules, mdl_relu_new());
  l_append_ptr( modules, mdl_dropout_new( drop_prob));
  l_append_ptr( modules, mdl_linear_new( hidden_dim, dim));

  if ( norm_type == MDL_BATCH_NORM_1D) {
    norm = (mdl_module) mdl_batch_norm1D_new( dim,
					      norm_eps,
					      norm_momentum);
  } else {
    assert( FALSE);
  }
  l_append_ptr( modules, norm);

  mdl_sequential seq1 = mdl_sequential_new( modules);
  assert( mdl_is_module( seq1));
  l_reset( modules);
  
  mdl_residual res = mdl_residual_new( (mdl_module) seq1);
  assert( mdl_is_module( res));
  l_append_ptr( modules, res);
  
  l_append_ptr( modules, mdl_relu_new());

  mdl_sequential seq2 = mdl_sequential_new( modules);
  assert( mdl_is_module( seq1));
  
  l_free( modules);
  
  return (mdl_module) seq2;  
  
}


static mdl_module build_net( const uint32_t img_sz,
			     const uint32_t hidden_dim,
			     const uint32_t num_blocks,
			     const uint32_t num_classes,
			     const float drop_prob,
			     const uint8_t norm_type,
			     const float norm_eps,
			     const float norm_momentum) {

  ag_init();
  // variables created during model building are either parameters
  // or module specific variables and are free'd there. We don't
  // have global list of these variables...
  ag_set_mode( AG_MDL_MODE);
  
  // these modules are copied into a sequential module and free'd there...
  l_list modules = l_new( 10, T_PTR, NULL);
  
  l_append_ptr( modules, mdl_linear_new( img_sz, hidden_dim));
  l_append_ptr( modules, mdl_relu_new( ));
  for ( int i = 0; i < num_blocks; i++) {
    l_append_ptr( modules, create_residual_block( hidden_dim,
						  (uint32_t) hidden_dim/2,
						  norm_type,
						  drop_prob,
						  norm_eps,
						  norm_momentum));
  }
  l_append_ptr( modules, mdl_linear_new( hidden_dim, num_classes));
  mdl_sequential seq = mdl_sequential_new( modules);

  l_free( modules);

  mdl_dump_modules( (mdl_module) seq, 0);
  
  return (mdl_module) seq;
}

static uint8_t check_params( const mdl_module model, const l_list params) {
  l_list pp = mdl_parameters( model, NULL);
  if ( l_len( pp) != l_len( params)) {
    fprintf( stderr, "check_params: lengths differ %d %d\n", l_len(pp), l_len( params));
    l_free( pp);
    return FALSE;
  }
  l_free( pp);

  for ( uint32_t i = 0 ; i < l_len( params); i++) {
    Value p = (Value) l_get_p( params, i);
    if ( !v_is_value( p)) {
      fprintf( stderr, "check_params: param %d not a value??\n", i);
      return FALSE;
    }
  }
  return TRUE;
}

static uint32_t get_max_idx_row( const t_tensor t, const uint32_t row_idx) {
  assert( RANK( t) == 2);

  uint32_t idx = 0;
  double m = t_2D_get( t, row_idx, 0);
  
  for ( uint32_t c = 1; c < T_N_COLS( t); c++) {
    const double v = t_2D_get( t, row_idx, c);
    if ( v > m) {
      idx = c;
      m = v;
    }
  }
  return idx;
}

static long compute_hits( const Value labels, const Value output) {
  assert( v_is_value( labels));
  assert( v_is_value( output));

  const t_tensor lbls = labels->data;
  const t_tensor out = output->data;

  assert( RANK( lbls) == 1);
  assert( RANK( out) == 2);

  assert( t_1D_len( lbls) == t_2D_nbr_rows( out));

  long hits = 0;

  for ( uint32_t i = 0; i < t_1D_len( lbls); i++) {
    uint32_t max_idx = get_max_idx_row( out, i);
    double l_i = t_1D_get( lbls, i);
    if ( ((uint32_t) l_i) == max_idx)
      hits++;
  }
  return hits;
}

static void nn_epoch( const mnist_data_loader data_loader,
		      const mdl_module model,
		      const uint8_t training
		      ) {
  
  // we have no global list for modules... the root module of model
  // recursively frees all nested modules. the loss_func though will be
  // disposed locally.
  mdl_softmax_loss loss_func = mdl_softmax_loss_new();

  if ( training) {

    mdl_train( model);

    // get model parameters for optimization. the parameters are Values which are *not*
    // kept in either the FWD or BWD list of values... they are persistent and thus must
    // be reset during gradient computations...
    l_list params = mdl_parameters( model, NULL);

    const float lr = 0.01;
    const float momentum = 0.0;
    const float weight_decay = 0.0;

    // params are passed to the optimizer and owned there...
    o_optimizer optim = (o_optimizer) o_sgd_new( params, lr, momentum, weight_decay);

    t_tensor labels = NULL;
    t_tensor images = NULL;
    long batch_cnt = 0;
    long total = 0;
    long hits = 0;
  
    while ( mnist_data_loader_next( data_loader,
				    &labels,
				    &images) >= 0) {

      log_msg( LOG_TRACE, "nn_epoch: batch %ld...\n", batch_cnt);
      batch_cnt++;
    
      ag_set_mode( AG_FWD_MODE);

      Value v_labels = v_tensor_to_value( labels, TRUE);
      Value v_images = v_tensor_to_value( images, TRUE);

      assert( v_is_tensor( v_images));
      assert( v_is_tensor( v_labels));
    
      Value output = mdl_call( model, 1, v_images);
      Value loss = mdl_call( (mdl_module) loss_func, 2, output, v_labels);

      // switches to BWD mode and fills the BWD value list
      ag_backward( loss, NULL, params);

      o_step( optim);
      o_reset_grad( optim);

      hits += compute_hits( v_labels, output);
      total += v_shape( v_labels)[0];
      
      l_reset( ag_get_val_list( AG_BWD_MODE));
      l_reset( ag_get_val_list( AG_FWD_MODE));

      t_free( labels);
      t_free( images);

      mem_dump_tbl( FALSE);

      float pct = ((float) hits)/((float) total) * 100.0;
      log_msg( LOG_NOTICE, "hits: %ld total: %ld pct: %f \%\n", hits, total, pct);
    
    }

    // the loss module is not part of the model...
    mdl_free( (mdl_module) loss_func);
  
    // the optimizer frees the params...
    o_free( optim);

  } else {  // not training

    mdl_eval( model);

    t_tensor labels = NULL;
    t_tensor images = NULL;
    long batch_cnt = 0;
    long total = 0;
    long hits = 0;
  
    while ( mnist_data_loader_next( data_loader,
				    &labels,
				    &images) >= 0) {

      log_msg( LOG_TRACE, "nn_epoch: batch %ld...\n", batch_cnt);
      batch_cnt++;

      ag_set_mode( AG_FWD_MODE);

      Value v_labels = v_tensor_to_value( labels, TRUE);
      Value v_images = v_tensor_to_value( images, TRUE);

      assert( v_is_tensor( v_images));
      assert( v_is_tensor( v_labels));
    
      Value output = mdl_call( model, 1, v_images);
      Value loss = mdl_call( (mdl_module) loss_func, 2, output, v_labels);

      hits += compute_hits( v_labels, output);
      total += v_shape( v_labels)[0];

      l_reset( ag_get_val_list( AG_FWD_MODE));

      t_free( labels);
      t_free( images);
      
      mem_dump_tbl( FALSE);

      float pct = ((float) hits)/((float) total) * 100.0;
      log_msg( LOG_NOTICE, "hits: %ld total: %ld pct: %f \%\n", hits, total, pct);
    }

    // the loss module is not part of the model...
    mdl_free( (mdl_module) loss_func);

  }
  
}

void main( int argc, char **argv) {

#define NN_BASIC
#ifdef NN_BASIC

  mnist_dataset dataset = mnist_dataset_new( MNIST_TRAIN_IMAGES, MNIST_TRAIN_LABELS, NULL);

  const uint8_t shuffle = TRUE; // FALSE;
  const uint32_t batch_size = 100;
  
  mnist_data_loader loader = mnist_data_loader_new( dataset, batch_size, shuffle);

  const uint32_t img_size = 28*28;
  const uint32_t hidden_dim = 100;
  const uint32_t num_blocks = 1; // 3
  const uint32_t num_classes = 10;
  const float drop_prob = 0.1;
  const float norm_eps = 1E-5;
  const float norm_momentum = 0.1;
  const uint8_t norm_type = MDL_BATCH_NORM_1D;
  
  mdl_module model = build_net( img_size, hidden_dim,
				num_blocks, num_classes, drop_prob,
				norm_type, norm_eps, norm_momentum);
  
#if 0
  t_tensor images = NULL;
  t_tensor labels = NULL;
  while ( mnist_data_loader_next( loader, &labels, &images) >= 0) {
    t_free( labels);
    t_free( images);
  }
#endif

  log_msg( LOG_TRACE, "training the model...\n");
  nn_epoch( loader, model, /* trainding */ TRUE);

  mnist_data_loader_reset( loader);
  
  log_msg( LOG_TRACE, "evaluating the model...\n");
  nn_epoch( loader, model, /* training */ FALSE);
  
  mnist_data_loader_free( loader);
  mdl_free( model);

  l_free( ag_get_val_list( AG_FWD_MODE));  
  l_free( ag_get_val_list( AG_BWD_MODE));
  //  l_free( ag_get_val_list( AG_MDL_MODE));
  
  mem_dump_tbl( TRUE);

  exit( 0);
#endif // NN_BASIC

#ifndef NN_BASIC
  
  // read the data
  t_tensor mnist_images = mnist_get_images( MNIST_TRAIN_IMAGES, TRUE);
  t_tensor mnist_labels = mnist_get_labels( MNIST_TRAIN_LABELS);

  mnist_images = mnist_flatten_images( mnist_images);
  
  assert( t_rank( mnist_images) == 2);
  assert( t_rank( mnist_labels) == 1);

  unsigned int y_max = mnist_get_y_max( mnist_labels);
    
  const double lr = 0.1;
  const unsigned int batch_size = 100;

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
  W_1 = t_div_scalar( W_1, sqrt( (double) hidden_dim), TRUE);
  W_2 = t_div_scalar( W_2, sqrt( (double) (y_max+1)), TRUE);

  ag_nn_epoch( mnist_images, mnist_labels, W_1, W_2, lr, batch_size, hidden_dim);

  t_free( W_1);
  t_free( W_2);

  t_free( mnist_images);
  t_free( mnist_labels);

  exit( 0);
#endif // !NN_BASIC

#if 0
  
  ag_init();
  
  double v1_data[1] = { 0};
  Value v1 = v_tensor( t_new_tensor( 0, NULL, T_FLOAT, v1_data));
  v_dump( v1, FALSE);
  Value v2 = v_exp( v1);
  v_dump( v2, FALSE);
  Value v3 = v_add_scalar( v2, 1);
  v_dump( v3, FALSE);
  Value v4 = v_mul( v2, v3);
  v_dump( v4, FALSE);
  ag_dump( ag_get_val_list( AG_FWD_MODE), FALSE);

  ag_gradient( v4, NULL);

  fprintf( stderr, "\n\n");
  ag_dump( ag_get_val_list( AG_FWD_MODE), FALSE);
  ag_dump( ag_get_val_list( AG_BWD_MODE), FALSE);

  l_free( ag_get_val_list( AG_FWD_MODE));
  l_free( ag_get_val_list( AG_BWD_MODE));

  exit( 0);

#endif
  
}
