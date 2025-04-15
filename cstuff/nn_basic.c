#include "mem.h"
#include "autograd.h"
#include "nn_basic.h"

#include <assert.h>
#include <string.h>

#define FALSE 0
#define TRUE 1

static uint8_t initialized = FALSE;

// keep record of all modules we create to free() them...
// static l_list module_list;

void mdl_free( mdl_module m) {
  // free parameters
  if ( m->parameters != NULL)
    l_free( m->parameters);
  
  // recurse over modules, depth-first
  if ( m->modules != NULL)
    l_free( m->modules);

  // clean up sub-classes as needed
  switch( m->sub_class) {
  case MDL_IDENTITY:
  case MDL_LINEAR:
  case MDL_FLATTEN:
  case MDL_RELU:
  case MDL_SEQUENTIAL:
  case MDL_SOFTMAX_LOSS:
    break;
  case MDL_BATCH_NORM_1D:
    {
      mdl_batch_norm1D b = (mdl_batch_norm1D) m;
      v_free( b->running_mean);
      v_free( b->running_var);
    }
    break;
  case MDL_LAYER_NORM_1D:
    {
      mdl_layer_norm1D b = (mdl_layer_norm1D) m;
    }
    break;
  case MDL_DROPOUT:
  case MDL_RESIDUAL:
    break;
  default:
    assert( FALSE);
  }
  MEM_FREE( m);
}

static void mdl_init() {
  if ( initialized)
    return;
  initialized = TRUE;
  // module_list = l_new( 10, T_PTR, (void *) mdl_free);
}

uint8_t mdl_is_module( const void *p) {
  assert( p != NULL);
  mdl_module m = (mdl_module) p;
  assert( m->sub_class != 0);
  return m->type_tag == MDL_TYPE_TAG;
}

static void init_mdl( const mdl_module m,
		      const uint16_t sub_class,
		      const uint16_t nbr_params,
		      const uint16_t nbr_modules) {

  mdl_init();
  
  m->type_tag = MDL_TYPE_TAG;
  m->sub_class = sub_class;

  if ( nbr_params > 0) {
    m->parameters = l_new( nbr_params, T_PTR, (void (*)) v_free);
  } else {
    m->parameters = NULL;
  }
  if ( nbr_modules > 0) {
    m->modules = l_new( nbr_modules, T_PTR, (void (*)) mdl_free);
  } else {
    m->modules = NULL;
  }
  m->training = FALSE;
  m->forward = NULL; // set by sub-class...

  // l_append_ptr( module_list, m);
}

static char *get_module_name( const mdl_module m) {
  switch( m->sub_class) {
  case MDL_IDENTITY:
    return "identity";
  case MDL_LINEAR:
    return "linear";
  case MDL_FLATTEN:
    return "flatten";
  case MDL_RELU:
    return "relu";
  case MDL_SEQUENTIAL:
    return "sequential";
  case MDL_SOFTMAX_LOSS:
    return "softmax_loss";
  case MDL_BATCH_NORM_1D:
    return "batch_norm_1D";
  case MDL_LAYER_NORM_1D:
    return "layer_norm_1D";
  case MDL_DROPOUT:
    return "dropout";
  case MDL_RESIDUAL:
    return "residual";
  default:
    assert( FALSE);
  }
}

void mdl_dump_modules( const mdl_module m, int level) {

  assert( mdl_is_module( m));
      
  char *mn = get_module_name( m);
  
  fprintf( stderr, "%*s%s\n", level, " ", mn);
  if ( m->modules == NULL)
    return;

  for ( int i = 0; i < l_len( m->modules); i++) {
    mdl_module c = (mdl_module) l_get_p( m->modules, i);
    mdl_dump_modules( c, level+2);
  }
}

// recursively traverse the children of module m and append to list
l_list mdl_children( const mdl_module m, l_list children) {

  assert( mdl_is_module( m));
  
  if ( m->modules == NULL)
    return children;

  // allocate children list
  if ( children == NULL)
    children = l_new( l_len( m->modules), T_PTR, NULL);

  for ( int i = 0; i < l_len( m->modules); i++) {
    mdl_module c = (mdl_module) l_get_p( m->modules, i);
    l_append_ptr( children, (void *) c);
    children = mdl_children( c, children);
  }
  return children;
}

l_list mdl_parameters( const mdl_module m, l_list parameters) {

  assert( mdl_is_module( m));

  if ( m->parameters == NULL && m->modules == NULL)
    return parameters;

  // allocate parameter list if needed
  if ( parameters == NULL && m->parameters != NULL) {
    parameters = l_new( l_len( m->parameters), T_PTR, NULL);
  }

  // append parameters
  if ( m->parameters != NULL) {
    assert( parameters != NULL);
    for ( int i = 0; i < l_len( m->parameters); i++) {
      Value c = (Value) l_get_p( m->parameters, i);
      assert( v_is_tensor( c));
      l_append_ptr( parameters, (void *) c);
    }
  }

  // append parameters of children recursively
  if ( m->modules != NULL) {
    for ( int i = 0; i < l_len( m->modules); i++) {
      mdl_module c = (mdl_module) l_get_p( m->modules, i);
      assert( mdl_is_module( c));
      parameters = mdl_parameters( c, parameters);
    }
  }

  return parameters;
  
}


static void mdl_set_training_flag( const mdl_module m, const uint8_t flag) {
  m->training = flag;
  l_list children = mdl_children( m, NULL);
  if ( children != NULL) {
    for ( uint32_t i = 0; i < l_len( children); i++) {
      mdl_module c = (mdl_module) l_get_p( children, i);
      c->training = flag;
    }
    l_free( children);
  }
}

void mdl_train( const mdl_module m) {
  assert( mdl_is_module( m));
  mdl_set_training_flag( m, TRUE);
}

void mdl_eval( const mdl_module m) {
  assert( mdl_is_module( m));
  mdl_set_training_flag( m, FALSE);
}

/*
  knowing which module we call, we have n_args and further params.
  mdl_module m = mdl_xxx_new(...)
  mdl_call( m, <n_args>, arg1, arg2 ... argn)
*/
Value mdl_call( const mdl_module m, uint32_t n_args, ...) {

  assert( mdl_is_module( m));
  assert( n_args == 1 || n_args == 2);

  // couldn't figure out how to pass var_args...

  Value vals[n_args];
  va_list args;
  va_start( args, n_args);
  for ( int i = 0; i < n_args; i++) {
    vals[i] = va_arg( args, Value);
    assert( ag_is_value( vals[i]));
  }
  va_end( args);

  Value t = NULL;
  
  switch( n_args) {
  case 1:
    t = (*m->forward) ((void *) m, n_args, vals[0]);
    break;
  case 2:
    t = (*m->forward) ((void *) m, n_args, vals[0], vals[1]);
    break;
  default:
    assert( FALSE);
  }
  assert( ag_is_value( t));
  return t;
}

static Value identity_fwd( const void * m, const uint32_t n_args, ...) {

  mdl_identity id = (mdl_identity) m;

  assert( mdl_is_module( id));
  assert( n_args == 1);

  // fprintf( stderr, "identity_fwd\n");
  
  va_list args;
  va_start( args, n_args);
  Value t = va_arg( args, Value);
  va_end( args);

  assert( ag_is_value( t));

  return t;
}

mdl_identity mdl_identity_new() {
  mdl_identity id = (mdl_identity) MEM_CALLOC( 1, sizeof( mdl_identity_struct));
  init_mdl( (mdl_module) id, MDL_IDENTITY, 0, 0);
  ((mdl_module) id)->forward = identity_fwd;
  return id;
}


static Value linear_fwd( const void *m, const uint32_t n_args, ...) {
  mdl_linear l = (mdl_linear) m;
  
  assert( n_args == 1);

  va_list args;
  va_start( args, n_args);
  const Value x = va_arg( args, Value);
  va_end( args);

  assert( ag_is_value( x));
  assert( mdl_is_module( l));
  assert( t_is2D( x->data));

  // fprintf( stderr, "linear_fwd\n");

  const uint32_t x_nbr_rows = t_2D_nbr_rows( x->data);
  const uint32_t x_nbr_cols = t_2D_nbr_cols( x->data);

  const uint32_t w_nbr_rows = t_2D_nbr_rows( l->weight->data);
  const uint32_t w_nbr_cols= t_2D_nbr_cols( l->weight->data);

  const uint32_t b_nbr_rows = t_2D_nbr_rows( l->bias->data);
  const uint32_t b_nbr_cols = t_2D_nbr_cols( l->bias->data);

  assert( x_nbr_cols == w_nbr_rows);
  assert( w_nbr_cols == b_nbr_cols);

  Value v1 = v_matmul( x, l->weight);
  // bias is (1, out_features). bcast to ( in_features, out_features)...
  Value bb = v_broadcast( l->bias, 2, v_shape( v1));
  Value v2 = v_add( v1, bb);

  return v2;
}

mdl_linear mdl_linear_new( const uint32_t in_features, const uint32_t out_features) {
  mdl_linear l = (mdl_linear) MEM_CALLOC( 1, sizeof( mdl_linear_struct));
  init_mdl( (mdl_module) l, MDL_LINEAR, 2, 0);
  ((mdl_module) l)->forward = linear_fwd;
  
  l->in_features = in_features;
  l->out_features = out_features;

  l->weight = v_kaiming_uniform( in_features, out_features, "RELU");
  l->bias = v_kaiming_uniform( 1, out_features, "RELU");
  assert( t_is2D( l->weight->data));
  assert( t_is2D( l->bias->data));

  l_append_ptr( l->mdl.parameters, l->weight);
  l_append_ptr( l->mdl.parameters, l->bias);

  return l;
}

static Value flatten_fwd( const void * m, const uint32_t n_args, ...) {

  mdl_flatten f = (mdl_flatten) m;
  
  assert( n_args == 1);
  
  va_list args;
  va_start( args, n_args);
  Value t = va_arg( args, Value);
  va_end( args);

  assert( mdl_is_module( f));
  assert( ag_is_value( t));

  // fprintf( stderr, "flatten_fwd\n");

  uint32_t shape[2];
  shape[0] = v_shape( t)[0];
  // build the size of all remaining dimensions shapes...
  shape[1] = 1;
  for ( int i = 1; i < v_rank(t); i++)
    shape[1] *= v_shape(t)[i];

  Value tt = v_reshape( t, 2, shape);
  return tt;
}


mdl_flatten mdl_flatten_new() {
  mdl_flatten f = (mdl_flatten) MEM_CALLOC( 1, sizeof( mdl_flatten_struct));
  init_mdl( (mdl_module) f, MDL_FLATTEN, 0, 0);
  ((mdl_module) f)->forward = flatten_fwd;
  return f;
}


static Value relu_fwd( const void * m, const uint32_t n_args, ...) {

  mdl_relu r = (mdl_relu) m;

  assert( mdl_is_module( r));
  assert( n_args == 1);
  
  va_list args;
  va_start( args, n_args);
  Value t = va_arg( args, Value);
  va_end( args);

  assert( ag_is_value( t));

  // fprintf( stderr, "relu_fwd\n");

  Value tt = v_relu( t);
  return tt;

}

mdl_relu mdl_relu_new() {
  mdl_relu r = (mdl_relu) MEM_CALLOC( 1, sizeof( mdl_relu_struct));
  init_mdl( (mdl_module) r, MDL_RELU, 0, 0);
  ((mdl_module) r)->forward = relu_fwd;
  return r;
}

static Value sequential_fwd( const void *m, const uint32_t n_args, ...) {
  mdl_sequential s = (mdl_sequential) m;

  assert( mdl_is_module( s));
  assert( n_args == 1);

  va_list args;
  va_start( args, n_args);
  Value t = va_arg( args, Value);
  va_end( args);

  assert( ag_is_value( t));

  // fprintf( stderr, "sequential_fwd\n");
  
  Value x_in = t;
  Value x_out = NULL;

    for ( uint32_t i = 0; i < l_len( s->mdl.modules); i++) {
    mdl_module mm = (mdl_module) l_get_p( s->mdl.modules, i);
    // the only forward() func with 2 args is softmax_loss...
    x_out = mm->forward( mm, 1, x_in); // mdl_call( mm, 1, x_in);
    assert( ag_is_value( x_out));
    x_in = x_out;
  }
  return x_out;
}

// creates a copy of the modules list...
mdl_sequential mdl_sequential_new( const l_list modules) {
  mdl_sequential s =  (mdl_sequential) MEM_CALLOC( 1, sizeof( mdl_sequential_struct));
  const uint32_t nbr_modules = l_len( modules);
  
  init_mdl( (mdl_module) s, MDL_SEQUENTIAL, 0, nbr_modules);

  for ( uint32_t i = 0; i < nbr_modules; i++) {
    mdl_module m = (mdl_module) l_get_p( modules, i);
    assert( mdl_is_module( m));
    l_append_ptr( s->mdl.modules, m);
  }

  ((mdl_module) s)->forward = sequential_fwd;
  return s;
}

static Value softmax_loss_fwd( const void *m, const uint32_t n_args, ...) {
  mdl_softmax_loss s = (mdl_softmax_loss) m;

  assert( n_args == 2);

  va_list args;
  va_start( args, n_args);
  Value logits = va_arg( args, Value);
  Value y = va_arg( args, Value);
  va_end( args);

  assert( mdl_is_module( s));
  assert( ag_is_value( logits));
  assert( ag_is_value( y));
  assert( t_is1D( y->data));
  assert( t_is2D( logits->data));

  // fprintf( stderr, "softmax_loss_fwd\n");
  

  // map 1D array of y-labels into a 2D one-hot matrix
  Value one_hot_y = v_one_hot( v_shape( logits)[1], y);
  const uint32_t len_y = v_shape( y)[0];

  // we select the max values of logits by multiplying with a one-hot matrix
  // and summing across columns *and* rows ... to get a scalar!!
  Value logits_div_len_y = v_div_scalar( logits, (double) len_y);
  Value one_hot_y_times_logits_div_len_y = v_mul( one_hot_y, logits_div_len_y);

  Value z_y = v_summation( one_hot_y_times_logits_div_len_y, 0, NULL, FALSE);
  assert( t_is0D( z_y->data));

  // log_sum_exp yields a 1D tensor, which is divided by a scalar
  // we indicate the axes here across which we perform the sum
  // column axes == 1 -> sum over rows...
  uint8_t axes[2] = {0,1};
  Value lse = v_log_sum_exp( logits, 2, axes);
  assert( t_is1D( lse->data));
  
  Value lse_div_len = v_div_scalar( lse, (double) len_y);
  assert( t_is1D( lse_div_len->data));

  // summation across all axes to yield a scalar...
  Value sum_lse = v_summation( lse_div_len, 0, NULL, FALSE);
  assert( t_is0D( sum_lse->data));

  Value res = v_sub( sum_lse, z_y);
  assert( t_is0D( res->data));

  return res;

}

mdl_softmax_loss mdl_softmax_loss_new() {
  mdl_softmax_loss s =  (mdl_softmax_loss) MEM_CALLOC( 1, sizeof( mdl_softmax_loss_struct));
  init_mdl( (mdl_module) s, MDL_SOFTMAX_LOSS, 0, 0);

  ((mdl_module) s)->forward = softmax_loss_fwd;
  return s;
}

static Value batch_norm1D_fwd( const void *m, const uint32_t n_args, ...) {
  mdl_batch_norm1D b = (mdl_batch_norm1D) m;

  assert( n_args == 1);

  va_list args;
  va_start( args, n_args);
  Value x = va_arg( args, Value);
  va_end( args);

  assert( mdl_is_module( b));
  assert( ag_is_value( x));

  // fprintf( stderr, "batch_norm1D_fwd\n");

  const uint32_t batch_size = v_shape(x)[0];
  const uint32_t features_size = v_shape(x)[1];

  uint8_t axes[2] = {1, 0}; // along columns: over all batches, i.e. axes=0
  Value mean = v_summation( x, 2, axes, TRUE);
  mean = v_div_scalar( mean, batch_size);
  Value x_minus_mean = v_sub( x, v_broadcast( mean, v_rank( x), v_shape( x)));
  Value var = v_div_scalar( v_summation( v_power_scalar( x_minus_mean, 2.0), 2, axes, TRUE), batch_size);

  if ( ((mdl_module) b)->training) {
    // this is horrible... due to memory management issues
    // we modify the data tensor of the Values in situ...
    t_tensor t_running_mean = b->running_mean->data;
    t_tensor t_running_var = b->running_var->data;

    // in-situ...
    t_mul_scalar( t_running_mean, (1.0-b->momentum), TRUE);
    // an identical copy
    t_tensor mean_data = t_squeeze( t_clone( mean->data), NULL);
    // in-situ multiply
    t_mul_scalar( mean_data, b->momentum, TRUE);
    // in-situ addition
    t_add( t_running_mean, mean_data, t_running_mean);
    t_free( mean_data);

    // in-situ...
    t_mul_scalar( t_running_var, (1.0-b->momentum), TRUE);
    // an identical copy
    t_tensor var_data = t_squeeze( t_clone( var->data), NULL);
    // in-situ multiply
    t_mul_scalar( var_data, b->momentum, TRUE);
    // in-situ addition
    t_add( t_running_var, var_data, t_running_var);
    t_free( var_data);

    /*
      ## use the formula from notebook to compute y
      ## the denominator...
      x_std = ((var + self.eps) ** 0.5).broadcast_to(x.shape)
      ## the nominator is x-mean...
      normed = x_minus_mean / x_std
      res = normed * self.weight.broadcast_to(x.shape) + self.bias.broadcast_to(x.shape)
    */
    // TBD
    Value x_std = v_broadcast( v_power_scalar( v_add_scalar( var, b->eps), 0.5),
			       v_rank( x), v_shape( x));
    Value normed = v_div( x_minus_mean, x_std);

    Value res = v_mul( normed, v_broadcast( b->weight, v_rank(x), v_shape(x)));
    res = v_add( res, v_broadcast( b->bias, v_rank(x), v_shape(x)));
    return res;
    
  } else {
    /*
      ## training == False
      normed = (x - self.running_mean) / (self.running_var + self.eps) ** 0.5
      res = normed * self.weight.broadcast_to(x.shape) + self.bias.broadcast_to(x.shape)
      return res
    */
    Value normed = v_sub( x, b->running_mean);
    Value v1 = v_power_scalar( v_add_scalar( b->running_var, b->eps), 0.5);
    normed = v_div( normed, v1);

    Value res = v_mul( normed, v_broadcast( b->weight, v_rank(x), v_shape(x)));
    res = v_add( res, v_broadcast( b->bias, v_rank(x), v_shape(x)));
    return res;
    
  }  
  
}

mdl_batch_norm1D mdl_batch_norm1D_new( const uint32_t nbr_channels,
				       const float eps,
				       const float momentum) {

  mdl_batch_norm1D b =  (mdl_batch_norm1D) MEM_CALLOC( 1, sizeof( mdl_batch_norm1D_struct));
  init_mdl( (mdl_module) b, MDL_BATCH_NORM_1D, 2, 0);

  ((mdl_module) b)->forward = batch_norm1D_fwd;

  b->nbr_channels = nbr_channels;
  b->eps = eps;
  b->momentum = momentum;

  uint32_t shape[1] = {nbr_channels};
  
  b->weight = v_ones( 1, shape);
  b->bias = v_zeros( 1, shape);

  l_append_ptr( b->mdl.parameters, b->weight);
  l_append_ptr( b->mdl.parameters, b->bias);

  // theses values are not parameters and thus must be freed explicitly and
  // not as part of freeing parameters...
  b->running_mean = v_zeros( 1, shape);
  b->running_var = v_ones( 1, shape);

  return b;

}
				   
static Value layer_norm1D_fwd( const void *m, const uint32_t n_args, ...) {
  mdl_layer_norm1D l = (mdl_layer_norm1D) m;

  assert( n_args == 1);

  va_list args;
  va_start( args, n_args);
  Value x = va_arg( args, Value);
  va_end( args);

  assert( mdl_is_module( l));
  assert( ag_is_value( x));

  // fprintf( stderr, "layer_norm1D_fwd\n");

  uint8_t axes[2] = {0,1}; // along rows
  Value mean = v_summation( x, 2, axes, TRUE);
  uint32_t shape[2] = {v_shape(x)[0], 1};
  mean = v_div_scalar( mean, v_shape(x)[1]);
  mean = v_reshape( mean, 2, shape);
  mean = v_broadcast( mean, v_rank( x), v_shape( x));

  Value var = v_summation( v_power_scalar( v_sub(x, mean), 2), 2, axes, TRUE);
  var = v_div_scalar( var, v_shape(x)[1]);
  var = v_reshape( var, 2, shape);
  var = v_broadcast( var, v_rank(x), v_shape(x));

  Value deno = v_add_scalar( var, l->eps);
  deno = v_power_scalar( deno, 0.5);

  Value res = v_broadcast( l->weight, v_rank(x), v_shape(x));
  Value x_mean_deno = v_div( v_sub( x, mean), deno);
  res = v_mul( res, x_mean_deno);
  res = v_add( res, v_broadcast( l->bias, v_rank(x), v_shape(x)));
  return res;
}

mdl_layer_norm1D mdl_layer_norm1D_new(  const uint32_t nbr_channels,
					const float eps) {

  mdl_layer_norm1D l =  (mdl_layer_norm1D) MEM_CALLOC( 1, sizeof( mdl_layer_norm1D_struct));
  init_mdl( (mdl_module) l, MDL_LAYER_NORM_1D, 2, 0);

  ((mdl_module) l)->forward = layer_norm1D_fwd;

  l->nbr_channels = nbr_channels;
  l->eps = eps;

  uint32_t shape[1] = {nbr_channels};
  
  l->weight = v_ones( 1, shape);
  l->bias = v_zeros( 1, shape);

  l_append_ptr( l->mdl.parameters, l->weight);
  l_append_ptr( l->mdl.parameters, l->bias);

  return l;

}


static Value dropout_fwd( const void *m, const uint32_t n_args, ...) {
  mdl_dropout l = (mdl_dropout) m;

  assert( n_args == 1);

  va_list args;
  va_start( args, n_args);
  Value x = va_arg( args, Value);
  va_end( args);

  // fprintf( stderr, "dropout_fwd\n");

  assert( mdl_is_module( l));
  assert( ag_is_value( x));

  if ( ((mdl_module)m)->training) {
    Value r = v_randb( v_rank(x), v_shape(x), (1.0-l->p));
    r = v_mul( x, r);
    r = v_div_scalar( x, (1.0-l->p));
    return r;
  } else {
    return x;
  }
  
}

mdl_dropout mdl_dropout_new( const float p) {
  mdl_dropout b =  (mdl_dropout) MEM_CALLOC( 1, sizeof( mdl_dropout_struct));
  init_mdl( (mdl_module) b, MDL_DROPOUT, 0, 0);

  ((mdl_module) b)->forward = dropout_fwd;

  b->p = p;

  return b;
}

static Value residual_fwd( const void *m, const uint32_t n_args, ...) {
  mdl_residual l = (mdl_residual) m;

  assert( n_args == 1);

  va_list args;
  va_start( args, n_args);
  Value x = va_arg( args, Value);
  va_end( args);

  assert( mdl_is_module( l));
  assert( mdl_is_module( l->fn));
  assert( ag_is_value( x));

  // fprintf( stderr, "residual_fwd\n");

  Value r = mdl_call( (const mdl_module) l->fn, 1, x);
  r = v_add( r, x);
  return r;
}

mdl_residual mdl_residual_new( const mdl_module fn) {
  mdl_residual b =  (mdl_residual) MEM_CALLOC( 1, sizeof( mdl_residual_struct));
  init_mdl( (mdl_module) b, MDL_RESIDUAL, 0, 1);

  assert( mdl_is_module( fn));

  l_append_ptr( b->mdl.modules, fn);

  b->fn = fn;

  ((mdl_module) b)->forward = residual_fwd;

  return b;
}

