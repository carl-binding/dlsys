#include "mem.h"
#include "nn_basic.h"

#include <assert.h>

#define FALSE 0
#define TRUE 1

static uint8_t initialized = FALSE;

// keep record of all modules we create to free() them...
static l_list module_list;

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
  case MDL_BATCH_NORM_1D:
  case MDL_LAYER_NORM_1D:
  case MDL_DROPOUT:
  case MDL_RESIDUAL:
  default:
    assert( FALSE);
  }
  MEM_FREE( m);
}

static void mdl_init() {
  if ( initialized)
    return;
  initialized = TRUE;
  module_list = l_new( 10, T_PTR, (void *) mdl_free);
}

uint8_t mdl_is_module( const void *p) {
  mdl_module m = (mdl_module) p;
  return m->type_tag == MDL_TYPE_TAG;
}

static void init_mdl( const mdl_module m, const uint16_t sub_class,
		      const uint16_t nbr_params, const uint16_t nbr_modules) {

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

  l_append_ptr( module_list, m);
}

// recursively traverse the children of module m and append to list
l_list mdl_children( const mdl_module m, l_list children) {

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
  mdl_set_training_flag( m, TRUE);
}

void mdl_eval( const mdl_module m) {
  mdl_set_training_flag( m, FALSE);
}

/*
  knowing which module we call, we have n_args and further params.
  mdl_module m = mdl_xxx_new(...)
  mdl_call( m, <n_args>, arg1, arg2 ... argn)
*/
Value mdl_call( const mdl_module m, uint32_t n_args, ...) {
  assert( mdl_is_module( m));
  va_list args;
  va_start( args, n_args);
  // just some sanity check...
  assert( n_args == 1 || n_args == 2);
  Value t = (*m->forward) ((void *) m, n_args, args);
  va_end( args);
  assert( v_is_tensor( t));
  return t;
}

static Value identity_fwd( const void * m, const uint32_t n_args, ...) {

  mdl_identity id = (mdl_identity) m;
  
  assert( n_args == 1);
  
  va_list args;
  va_start( args, n_args);
  Value t = va_arg( args, Value);
  va_end( args);

  assert( mdl_is_module( id));
  assert( v_is_tensor( t));

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

  assert( v_is_tensor( x));
  assert( mdl_is_module( l));
  assert( t_is2D( x->data));

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
  Value bb = v_broadcast( l->bias, 2, v1->data->shape);
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
}

static Value flatten_fwd( const void * m, const uint32_t n_args, ...) {

  mdl_flatten f = (mdl_flatten) m;
  
  assert( n_args == 1);
  
  va_list args;
  va_start( args, n_args);
  Value t = va_arg( args, Value);
  va_end( args);

  assert( mdl_is_module( f));
  assert( v_is_tensor( t));

  uint32_t shape[2];
  shape[0] = t->data->shape[0];
  // build the size of all remaining dimensions shapes...
  shape[1] = 1;
  for ( int i = 1; i < t->data->rank; i++)
    shape[1] *= t->data->shape[i];

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
  
  assert( n_args == 1);
  
  va_list args;
  va_start( args, n_args);
  Value t = va_arg( args, Value);
  va_end( args);

  assert( mdl_is_module( r));
  assert( v_is_tensor( t));

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

  assert( n_args == 1);

  va_list args;
  va_start( args, n_args);
  Value t = va_arg( args, Value);
  va_end( args);

  assert( mdl_is_module( s));
  assert( v_is_tensor( t));

  Value x_in = t;
  Value x_out = NULL;

  for ( uint32_t i = 0; i < l_len( s->mdl.modules); i++) {
    mdl_module mm = (mdl_module) l_get_p( s->mdl.modules, i);
    x_out = mm->forward( mm, 1, x_in); // mdl_call( mm, 1, x_in);
    x_in = x_out;
  }
  return x_out;
}


mdl_sequential mdl_sequential_new( const uint32_t n_args, ...) {
  mdl_sequential s =  (mdl_sequential) MEM_CALLOC( 1, sizeof( mdl_sequential_struct));
  init_mdl( (mdl_module) s, MDL_SEQUENTIAL, 0, n_args);

  va_list args;
  va_start( args, n_args);
  for ( uint32_t i = 0; i < n_args; i++) {
    mdl_module m = va_arg( args, mdl_module);
    assert( mdl_is_module( m));
    l_append_ptr( s->mdl.modules, m);
  }
  va_end( args);

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
  assert( v_is_tensor( logits));
  assert( v_is_tensor( y));
  assert( t_is1D( y->data));
  assert( t_is2D( logits->data));

  // map 1D array of y-labels into a 2D one-hot matrix
  Value one_hot_y = v_one_hot( logits->data->shape[1], y);
  const uint32_t len_y = y->data->shape[0];

  // we select the max values of logits by multiplying with a one-hot matrix
  // and summing across columns *and* rows ... to get a scalar!!
  Value logits_div_len_y = v_div_scalar( logits, (double) len_y);
  Value one_hot_y_times_logits_div_len_y = v_mul( one_hot_y, logits_div_len_y);
  Value z_y = v_summation( one_hot_y_times_logits_div_len_y, 0, NULL, FALSE);
  assert( t_is0D( z_y->data));

  // log_sum_exp yields a 1D tensor, which is divided by a scalar
  uint8_t axes[2] = {0,1};
  Value lse = v_div_scalar( v_log_sum_exp( logits, 2, axes), (double) len_y);
  assert( t_is1D( lse->data));

  // summation across all axes to yield a scalar...
  Value sum_lse = v_summation( lse, 0, NULL, FALSE);
  assert( t_is0D( sum_lse->data));

  Value res = v_sub( sum_lse, z_y);
  assert( t_is0D( res->data));

  return res;

}

mdl_softmax_loss mdl_softmax_loss_new( const uint32_t n_args, ...) {
  mdl_softmax_loss s =  (mdl_softmax_loss) MEM_CALLOC( 1, sizeof( mdl_softmax_loss_struct));
  init_mdl( (mdl_module) s, MDL_SOFTMAX_LOSS, 0, n_args);

  assert( n_args == 0);

  ((mdl_module) s)->forward = softmax_loss_fwd;
  return s;
}
