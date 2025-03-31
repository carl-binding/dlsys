#include <math.h>
#include <assert.h>
#include <string.h>

#include "mem.h"
#include "autograd.h"
#include "tensor.h"
#include "mnist.h"


#define FALSE 0
#define TRUE 1

// we keep three lists of values: one for fwd values, one for bwd (gradient) values, and one for (static) values in modules
static l_list fwd_val_list = NULL;
static l_list bwd_val_list = NULL;
static l_list mdl_val_list = NULL;

static uint8_t val_mode = AG_FWD_MODE;

uint8_t ag_get_mode() {
  return val_mode;
}

void ag_set_mode( const uint8_t m) {
  assert( m >= AG_FWD_MODE && m <= AG_MDL_MODE);
  val_mode = m;
}

l_list ag_get_val_list( const uint8_t mode) {
  ag_init();

  l_list l = NULL;
  
  switch( mode) {
  case AG_FWD_MODE:
    l = fwd_val_list;
    break;
  case AG_BWD_MODE:
    l = bwd_val_list;
    break;
  case AG_MDL_MODE:
    l = mdl_val_list;
    break;
  default:
    assert( FALSE);
  }
  return l;
}

static uint8_t initialized = FALSE;
static uint8_t debug = TRUE;

// to allocate axes given the rank and the values of axes which are
// 0 or 1...
static uint8_t *alloc_axes( const uint16_t rank, const t_axes axes) {
  if ( rank == 0) {
    assert( axes == NULL);
    return NULL;
  }
  
  assert( axes != NULL);
  uint8_t *a = (uint8_t *) MEM_CALLOC( rank, sizeof( uint8_t));
  memcpy( a, axes, rank*sizeof( uint8_t));

  return a;
}

void v_free( void *_v);

void ag_init() {
  if ( initialized)
    return;

  fwd_val_list = l_new( 10, T_PTR, v_free);
  bwd_val_list = l_new( 10, T_PTR, v_free);
  mdl_val_list = l_new( 10, T_PTR, v_free);

  initialized = TRUE;
  
}

uint8_t ag_is_value( const void *p) {
  Value v = (Value) p;
  return v->type_tag == AG_VALUE_TYPE_TAG;
}

uint16_t v_rank( const Value v) {
  return v->data->rank;
}

uint32_t *v_shape( const Value v) {
  return v->data->shape;
}

uint8_t v_dtype( const Value v) {
  return v->data->dtype;
}


static Value value_new() {
  Value v = (Value) MEM_CALLOC( 1, sizeof( Value_struct));
  v->type_tag = AG_VALUE_TYPE_TAG;

  l_list l = NULL;
  switch( val_mode) {
  case AG_FWD_MODE:
    l = fwd_val_list;
    break;
  case AG_BWD_MODE:
    l = bwd_val_list;
    break;
  case AG_MDL_MODE:
    l = mdl_val_list;
    break;
  default:
    assert( FALSE);
  }

  l_append_ptr( l, (const void *) v);

  return v;
}

static l_list alloc_inputs( const int n_args,
			    ...) {
  l_list l = l_new( n_args, T_PTR, NULL);

  va_list argptr;
  va_start( argptr, n_args);
  
  for ( unsigned int i = 0; i < n_args; i++) {
    const Value v = va_arg( argptr, Value);
    l_append_ptr( l, (const void *) v);
  }
  va_end( argptr);
  return l;
}

uint8_t ag_is_op( const void *p) {
  Op o = (Op) p;
  return o->type_tag == AG_OP_TYPE_TAG;
}

Op op_new(
	  t_tensor (*compute) (const void *self, const uint32_t n_args, const l_list args),
	  void * (*gradient) (const void *self, 
			      const uint32_t k_idx,
			      const void *v_i),
	  const uint16_t n_args_compute
	  ) {
  Op op = (Op) MEM_CALLOC( 1, sizeof( Op_struct));
  op->type_tag = AG_OP_TYPE_TAG;
  op->compute = compute;
  op->gradient = gradient;
  op->n_args_compute = n_args_compute;
  return op;
}

void op_free( Op op) {

  if ( op == NULL)
    return;
  
  switch ( op->sub_type) {
  case AG_ADD_SCALAR:
  case AG_MUL_SCALAR:
  case AG_DIV_SCALAR:
  case AG_POWER_SCALAR:
    MEM_FREE( op->u.s);
    break;
  case AG_TRANSPOSE:
  case AG_SUMMATION:
  case AG_LOG_SUM_EXP:
    MEM_FREE( op->u.axes.axes);
    break;
  case AG_RESHAPE:
  case AG_BROADCAST_TO:
    MEM_FREE( op->u.shape.shape);
    break;
  default:
    break;
  }

  MEM_FREE( op);
}

// convienence to get a value out of a list of args which contains Values
static Value get_val( const l_list l, const unsigned int idx) {
  l_el el = l_get( l, idx);
  const Value v = (Value) el.ptr;
  return v;
}

// convienence to get a tensor of a list of args containing Tensor-Values
static t_tensor get_arg( const l_list args, const unsigned int idx) {
  const Value v = get_val( args, idx);
  assert( v != NULL && v->data != NULL);
  return v->data;
}


// =========================== EWiseAdd =================================================

static void *ewise_add_gradient(  const void *_self,
				  const uint32_t k_idx,
				  const void *_v_i) {

  assert( k_idx == 0 || k_idx == 1); // two operands

  const Op self = (const Op) _self;
  const Value v_i = (Value) _v_i; // node i in forward evaluation
  assert( v_i->adjoint != NULL);

  // v = a + b, dv/da = 1, dv/db = 1, thus adj*dv/db = adj*dv/da = adj
  return v_i->adjoint;  // out_grad...
}

static t_tensor ewise_add( const void *self,
			   const uint32_t n_args,
			   const l_list args) {
  assert( n_args == 2);
  const t_tensor t1 = get_arg( args, 0);
  const t_tensor t2 = get_arg( args, 1);
  return t_add( t1, t2, NULL);
}

Op EWiseAdd_new() {
  Op op = op_new( ewise_add, ewise_add_gradient, 2);
  op->sub_type = AG_EWISE_ADD;
  return op;
}  

Value v_add( const Value v1, const Value v2) {
  Value v = value_new();
  v->op = EWiseAdd_new();
  v->inputs = alloc_inputs( 2, v1, v2);
  v->data = (v->op->compute)(v->op, 2, v->inputs);
  return v;
}

// ==================================== EWiseSub =================

static void *ewise_sub_gradient(  const void *_self,
				  const uint32_t k_idx,
				  const void *_v_i) {
  assert( k_idx == 0 || k_idx == 1);
  
  const Op self = (const Op) _self;
  assert( self->n_args_compute == 2);
  assert( k_idx < 2);

  const Value v_i = (Value) _v_i; // node i in forward evaluation
  assert( v_i->adjoint != NULL);

  // v = a - b, dv/da = 1, dv/db = -1, thus adj*dv/db = -adj, adj*dv/da = adj

  if ( k_idx == 0) // dv/da
    return v_i->adjoint;
  else // dv/db
    return v_negate( v_i->adjoint);
}

static t_tensor ewise_sub( const void *self,
			   const uint32_t n_args,
			   const l_list args) {
  assert( n_args == 2);
  const t_tensor t1 = get_arg( args, 0);
  const t_tensor t2 = get_arg( args, 1);
  return t_subtract( t1, t2, NULL);
}

Op EWiseSub_new() {
  Op op = op_new( ewise_sub, ewise_sub_gradient, 2);
  op->sub_type = AG_EWISE_SUB;
  return op;
}  

Value v_sub( const Value v1, const Value v2) {
  Value v = value_new();
  v->op = EWiseSub_new();
  v->inputs = alloc_inputs( 2, v1, v2);
  v->data = (v->op->compute)(v->op, 2, v->inputs);
  return v;
}

// ==================================  EWiseMul =========================

static void *ewise_mul_gradient( const void *_self,
				 const uint32_t k_idx,
				 const void *_v_i) {
  assert( k_idx <= 1);
  
  const Op self = (const Op) _self;

  const Value v_i = (Value) _v_i; // node i in forward evaluation
  assert( v_i->adjoint != NULL);

  assert( k_idx < v_i->inputs->cnt);
  assert( self->n_args_compute == 2);

  // v = a * b, dv/da = b, dv/db = a : we flip the args...
  const uint32_t kk_idx = (k_idx==0)?1:0;
  const Value other_arg = get_val( v_i->inputs, kk_idx);

  // elementwise multiply
  Value v_bar_k_i = v_mul( v_i->adjoint, other_arg);

  return v_bar_k_i;
}

static t_tensor ewise_mul( const void *self,
			   const uint32_t n_args,
			   const l_list args) {
  assert( n_args == 2);
  const t_tensor t1 = get_arg( args, 0);
  const t_tensor t2 = get_arg( args, 1);
  return t_multiply( t1, t2, NULL);
}

Op EWiseMul_new() {
  Op op = op_new( ewise_mul, ewise_mul_gradient, 2);
  op->sub_type = AG_EWISE_MUL;
  return op;
}  

Value v_mul( const Value v1, const Value v2) {
  Value v = value_new();
  v->op = EWiseMul_new();
  v->inputs = alloc_inputs( 2, v1, v2);
  v->data = (v->op->compute)(v->op, 2, v->inputs);
  return v;
}


// ================================== EWiseDiv ===========================

static void *ewise_div_gradient( const void *_self,
				 const uint32_t k_idx,
				 const void *_v_i) {
  const Op self = (const Op) _self;

  const Value v_i = (Value) _v_i; // node i in forward evaluation
  assert( v_i->adjoint != NULL);

  assert( k_idx < v_i->inputs->cnt);
  assert( self->n_args_compute == 2);

  const Value a = get_val( v_i->inputs, 0);
  const Value b = get_val( v_i->inputs, 1);

  Value v_bar_k_i = NULL;

  // v = a / b, dv/da = 1/b, dv/db = -a/b^2
  if ( k_idx == 0) { // dv/da
    v_bar_k_i = v_div( v_i->adjoint, b);
  } else { // dv/db
    assert( k_idx == 1);
    const Value b_squared = v_mul( b, b);
    const Value b_squared_negated = v_negate( b_squared);
    const Value a_over_b_squared_negated = v_div( a, b_squared_negated);
    // v_bar_i * dv/db
    v_bar_k_i = v_mul( v_i->adjoint, a_over_b_squared_negated);
  }

  return v_bar_k_i;
}

static t_tensor ewise_div( const void *self,
			   const uint32_t n_args,
			   const l_list args) {
  assert( n_args == 2);
  const t_tensor t1 = get_arg( args, 0);
  const t_tensor t2 = get_arg( args, 1);
  return  t_divide( t1, t2, NULL);
}

Op EWiseDiv_new() {
  Op op = op_new( ewise_div, ewise_div_gradient, 2);
  op->sub_type = AG_EWISE_DIV;
  return op;
}  

Value v_div( const Value v1, const Value v2) {
  Value v = value_new();
  v->op = EWiseDiv_new();
  v->inputs = alloc_inputs( 2, v1, v2);
  v->data = (v->op->compute)(v->op, 2, v->inputs);
  return v;
}

// ================================ EWisePower ==========================

// x^y element-wise i.e. x_00^y_00, etc.
static t_tensor ewise_power( const void *self,
			   const uint32_t n_args,
			   const l_list args) {
  assert( n_args == 2);
  const t_tensor t1 = get_arg( args, 0);
  const t_tensor t2 = get_arg( args, 1);
  assert( t_same_shape( t1, t2));
  return t_power( t1, t2, NULL);
}

static void *ewise_power_gradient(const void *_self,
				 const uint32_t k_idx,
				 const void *_v_i) {
  const Op self = (const Op) _self;

  const Value v_i = (Value) _v_i; // node i in forward evaluation
  assert( v_i->adjoint != NULL);

  assert( k_idx < v_i->inputs->cnt);
  assert( self->n_args_compute == 2);

  const Value a = get_val( v_i->inputs, 0);
  const Value b = get_val( v_i->inputs, 1);

  Value v_bar_k_i = NULL;

  // d(a^b)/da = b * a^(b-1), d(a^b)/db = a^b * ln( a)
  if ( k_idx == 0) {  // grad_a
    // out_grad * b
    Value v1 = v_mul( v_i->adjoint, b);
    // power( a, b-1)
    Value b_minus_one = v_add_scalar( b, -1);
    Value v2 = v_power( a, b_minus_one);
    // out_grad * b * power( a, b-1)
    v_bar_k_i = v_mul( v1, v2);      
  } else { // grad_b
    assert( k_idx == 1);
    // out_grad * node
    Value v1 = v_mul( v_i->adjoint, v_i);
    // log( a)
    Value v2 = v_log( a);
    // out_grad * node * log(a)
    v_bar_k_i = v_mul( v1, v2);
  }
  return v_bar_k_i;
}

Op EWisePow_new() {
  Op op = op_new( ewise_power, ewise_power_gradient, 2);
  op->sub_type = AG_EWISE_POW;
  return op;
}  

Value v_power( const Value u, const Value w) {
  Value v = value_new();
  v->op = EWisePow_new();
  v->inputs = alloc_inputs( 2, u, w);
  v->data = (v->op->compute)(v->op, 2, v->inputs);
  return v;
}

// =========================== AddScalar ===================================

static t_tensor add_scalar( const void *_self,
			    const uint32_t n_args,
			    const l_list args) {
  const Op self = (const Op) _self;
  assert( n_args == 1);
  assert( self->sub_type == AG_ADD_SCALAR);
  const t_tensor t1 = get_arg( args, 0);
  return t_add( t1, self->u.s, NULL);
}

static void *add_scalar_gradient( const void *_self,
				  const uint32_t k_idx,
				  const void *_v_i) {
  const Op self = (const Op) _self;

  const Value v_i = (Value) _v_i; // node i in forward evaluation

  assert( k_idx < v_i->inputs->cnt);
  assert( self->n_args_compute == 1);

  // v = x + s, dv/dx = 1, multiplied with gradient == adjoint...
  return v_i->adjoint;
}

Op AddScalar_new( const double s) {
  // we use t_add which works on 0-dim tensor being added to n-dim tensor...
  Op op = op_new( add_scalar, add_scalar_gradient, 1);
  op->sub_type = AG_ADD_SCALAR;
  op->u.s = t_new_scalar( s, T_FLOAT);
  return op;
}

Value v_add_scalar( const Value v1, const double s) {
  Value v = value_new();
  v->op = AddScalar_new( s); // turns scalar into tensor
  v->inputs = alloc_inputs( 1, v1);
  v->data = (v->op->compute)(v->op, 1, v->inputs);
  return v;
}

// =========================================== MulScalar ==========================
static t_tensor mul_scalar( const void *_self,
			    const uint32_t n_args,
			    const l_list args) {
  const Op self = (const Op) _self;
  assert( n_args == 1);
  assert( self->sub_type == AG_MUL_SCALAR);
  const t_tensor t1 = get_arg( args, 0);
  return t_multiply( t1, self->u.s, NULL);
}

static void *mul_scalar_gradient(const void *_self,
				 const uint32_t k_idx,
				 const void *_v_i) {
  const Op self = (const Op) _self;

  const Value v_i = (Value) _v_i; // node i in forward evaluation
  assert( k_idx == 0);
  assert( v_i->inputs->cnt == 1);

  assert( self->sub_type == AG_MUL_SCALAR);

  // v = s * x, dv/dx = s
  assert( self->u.s->rank == 1);
  const double sv = t_scalar( self->u.s);

  // v_bar_k_i = adj_i * dv_i/dv_k
  Value v_bar_k_i = v_mul_scalar( v_i->adjoint, sv);

  return v_bar_k_i;
  
}

Op MulScalar_new( const double s) {
  // we use t_multiply which works on 0-dim tensor being multiplied with n-dim tensor...
  Op op = op_new( mul_scalar, mul_scalar_gradient, 1);
  op->sub_type = AG_MUL_SCALAR;
  op->u.s = t_new_scalar( s, T_FLOAT);
  return op;
}

Value v_mul_scalar( const Value v1, const double s) {
  Value v = value_new();
  v->op = MulScalar_new( s); // turns scalar into tensor
  v->inputs = alloc_inputs( 1, v1);
  v->data = (v->op->compute)(v->op, 1, v->inputs);
  return v;
}

// ========================== DivScalar ======================================


static t_tensor div_scalar( const void *_self,
			     const uint32_t n_args,
			    const l_list args) {
  const Op self = (const Op) _self;
  assert( n_args == 1);
  assert( self->sub_type == AG_DIV_SCALAR);

  const t_tensor t1 = get_arg( args, 0);
  return t_divide( t1, self->u.s, NULL);
}

static void *div_scalar_gradient( const void *_self,
				  const uint32_t k_idx,
				  const void *_v_i) {
  const Op self = (const Op) _self;

  const Value v_i = (Value) _v_i; // node i in forward evaluation

  assert( k_idx < v_i->inputs->cnt);
  assert( self->n_args_compute == 1);

  // v = x/s, dv/dx = 1/s
  assert( self->u.s->rank == 1);
  const double sv = t_scalar( self->u.s);

  // divide_scalar( out_grad, self.scalar): we multiply by 1/self.scalar...
  Value v_bar_k_i = v_mul_scalar( v_i->adjoint, 1.0/sv);

  return v_bar_k_i;
  
}

Op DivScalar_new( const double s) {
  // we use t_divide which works on 0-dim tensor divising an n-dim tensor...
  Op op = op_new( div_scalar, div_scalar_gradient, 1);
  op->sub_type = AG_DIV_SCALAR;
  op->u.s = t_new_scalar( s, T_FLOAT);
  return op;
}

Value v_div_scalar( const Value v1, const double s) {
  Value v = value_new();
  v->op = DivScalar_new( s); // turns scalar into tensor
  v->inputs = alloc_inputs( 1, v1);
  v->data = (v->op->compute)(v->op, 1, v->inputs);
  return v;
}

// ============================ PowerScalar ================

// x^s
static t_tensor power_scalar( const void *_self,
			     const uint32_t n_args,
			    const l_list args) {
  const Op self = (const Op) _self;
  assert( n_args == 1);
  assert( self->sub_type == AG_POWER_SCALAR);
  const t_tensor t1 = get_arg( args, 0);
  return t_power( t1, self->u.s, NULL);
}

static void *power_scalar_gradient( const void *_self,
				  const uint32_t k_idx,
				  const void *_v_i) {
  const Op self = (const Op) _self;

  const Value v_i = (Value) _v_i; // node i in forward evaluation

  assert( k_idx < v_i->inputs->cnt);
  assert( self->n_args_compute == 1);

  const Value v_k = get_val( v_i->inputs, 0);
  
  // v = x^s, dv/dx = s*x^(s-1)
  assert( self->u.s->rank == 1);
  const double sv = t_scalar( self->u.s);

  Value x_power_s_minus_one = v_power_scalar( v_k, sv-1.0);
  Value s_times_x_power_s_minus_one = v_mul_scalar( x_power_s_minus_one, sv);
  Value v_bar_k_i = v_mul( v_i->adjoint, s_times_x_power_s_minus_one);

  return v_bar_k_i;
  
}

Op PowerScalar_new( const double s) {
  Op op = op_new( power_scalar, power_scalar_gradient, 1);
  op->sub_type = AG_POWER_SCALAR;
  op->u.s = t_new_scalar( s, T_FLOAT);
  return op;
}

Value v_power_scalar( const Value v1, const double s) {
  Value v = value_new();
  v->op = PowerScalar_new( s); // turns scalar into tensor
  v->inputs = alloc_inputs( 1, v1);
  v->data = (v->op->compute)(v->op, 1, v->inputs);
  return v;
}

// ======================= Transpose ================================

static t_tensor transpose_tensor( const void *_self,
				  const uint32_t n_args,
				  const l_list args) {
  const Op self = (const Op) _self;
  assert( n_args == 1);
  assert( self->sub_type == AG_TRANSPOSE);

  const t_tensor t1 = get_arg( args, 0);
  assert( t1 != NULL);
  
  t_tensor tt = t_transpose_axes( t1, self->u.axes.axes_len, self->u.axes.axes);
  assert( tt != NULL);

  return tt;
}

static void *transpose_gradient(const void *_self,
				const uint32_t k_idx,
				const void *_v_i) {
  const Op self = (const Op) _self;

  const Value v_i = (Value) _v_i; // node i in forward evaluation

  assert( v_i->data != NULL);  // v_i is the output node
  assert( k_idx < v_i->inputs->cnt);
  assert( self->n_args_compute == 1);

  // C = tranpose( A)
  // A_bar = inv_transpose( C_bar), i.e. grad_in_node = inv_transpose( out_grad)
  // and the inverse of the tranpose is the tranpose itself...
  Value v_bar_k_i = v_transpose( v_i->adjoint, self->u.axes.axes_len, self->u.axes.axes);

  return v_bar_k_i;
}

Op Transpose_new( const uint16_t axes_len, const t_axes axes) {
  Op op = op_new( transpose_tensor, transpose_gradient, 1);
  op->sub_type = AG_TRANSPOSE;
  if ( axes_len > 0) {
    assert( axes != NULL);
    op->u.axes.axes_len = axes_len;
    op->u.axes.axes = alloc_axes( axes_len, axes);
  } else {
    assert( axes == NULL);
    op->u.axes.axes_len = 0;
    op->u.axes.axes = NULL;
  }    
  return op;
}

Value v_transpose( const Value v1, const uint16_t axes_len, const t_axes axes) {
  Value v = value_new();
  v->op = Transpose_new( axes_len, axes);
  v->inputs = alloc_inputs( 1, v1);
  v->data = (v->op->compute) (v->op, 1, v->inputs);
  assert( v->data != NULL);
  return v;
}

// ===================================== Summation ============================
static t_tensor summation_tensor( const void *_self,
			    const uint32_t n_args,
			    const l_list args) {
  const Op self = (const Op) _self;
  assert( n_args == 1);
  assert( self->sub_type == AG_SUMMATION);

  const t_tensor t1 = get_arg( args, 0);

  assert( t1->rank == self->u.axes.axes_len);
  return t_sum( t1, self->u.axes.axes, self->keep_dims);
}

static void *summation_gradient(const void *_self,
				const uint32_t k_idx,
				const void *_v_i) {
  const Op self = (const Op) _self;
  assert( self->sub_type == AG_SUMMATION);

  const Value v_i = (Value) _v_i; // node i in forward evaluation

  assert( v_i->data != NULL);  // v_i is the output node
  assert( k_idx < v_i->inputs->cnt);
  assert( self->n_args_compute == 1);

  const Value v_k = get_val( v_i->inputs, 0);
  /**
      # Broadcast out_grad to the input shape, i.e. the
      # smaller out-grad tensor to the larger input shape.
      ## courtesy of ChatGPT
      grad_input = broadcast_to(out_grad, input.shape)
      return grad_input
  */

  const Value out_grad = (Value) v_i->adjoint;
  const uint16_t in_shape_len = v_k->data->rank;
  const uint32_t *in_shape = v_k->data->shape;
  return v_broadcast( out_grad, in_shape_len, in_shape);
}


Op Summation_new(  const uint16_t axes_len, const t_axes axes, const uint8_t keep_dims) {
  Op op = op_new( summation_tensor, summation_gradient, 1);
  op->sub_type = AG_SUMMATION;
  if ( axes_len == 0) {
    assert( axes == NULL);
    op->u.axes.axes_len = 0;
    op->u.axes.axes = NULL;    
  } else {
    assert( axes != NULL);
    op->u.axes.axes_len = axes_len;
    op->u.axes.axes = alloc_axes( axes_len, axes);
  }
  op->keep_dims = keep_dims;
  return op;
}

Value v_summation( const Value v, const uint16_t rank, const t_axes axes, const uint8_t keep_dims) {
  Value vv = value_new();
  vv->op = Summation_new( rank, axes, keep_dims);
  vv->inputs = alloc_inputs( 1, v);
  vv->data = (vv->op->compute)(vv->op, 1, vv->inputs);
  return vv;
}

// =====================================  Reshape ===============================

static t_tensor reshape_tensor( const void *_self,
			    const uint32_t n_args,
			    const l_list args) {
  const Op self = (const Op) _self;
  assert( n_args == 1);
  assert( self->sub_type == AG_RESHAPE);

  const t_tensor t1 = get_arg( args, 0);

  assert( t1->rank == self->u.shape.rank);
  return t_reshape( t1, self->u.shape.rank, self->u.shape.shape, NULL);
}

static uint32_t *alloc_shape( const uint16_t rank, const t_shape shape) {
  uint32_t *s = (uint32_t *) MEM_CALLOC( rank, sizeof( uint32_t));
  memcpy( s, shape, rank*sizeof( uint32_t));
  return s;
}

static void *reshape_gradient(const void *_self,
				const uint32_t k_idx,
				const void *_v_i) {
  const Op self = (const Op) _self;

  const Value v_i = (Value) _v_i; // node i in forward evaluation

  assert( v_i->data != NULL);  // v_i is the output node
  assert( k_idx < v_i->inputs->cnt);
  assert( self->n_args_compute == 1);

  // A_bar = inv_reshape( C_bar), i.e. grad_in_node = inv_reshape( out_grad)
  // and the inverse of the reshape is the reshape to the original shape...
  return v_reshape( v_i->adjoint, self->u.shape.rank, self->u.shape.shape);

}

Op Reshape_new( const uint16_t rank, const t_shape shape) {
  Op op = op_new( reshape_tensor, reshape_gradient, 1);
  op->sub_type = AG_RESHAPE;
  op->u.shape.rank = rank;
  op->u.shape.shape = alloc_shape( rank, shape);
  return op;
}

Value v_reshape( const Value v, const uint16_t rank, const t_shape shape) {
  assert( FALSE);
  return NULL;
}

// ============================== Broadcast ================================

static t_tensor broadcast_tensor( const void *_self,
			    const uint32_t n_args,
			    const l_list args) {
  const Op self = (const Op) _self;
  assert( n_args == 1);
  assert( self->sub_type == AG_BROADCAST_TO);

  const t_tensor t1 = get_arg( args, 0);

  assert( t1->rank == self->u.shape.rank);
  return t_broadcast_to( t1, self->u.shape.rank, self->u.shape.shape);
}

static void *broadcast_gradient(const void *_self,
				const uint32_t k_idx,
				const void *_v_i) {
  const Op self = (const Op) _self;
  assert( self->sub_type == AG_BROADCAST_TO);

  const Value v_i = (Value) _v_i; // node i in forward evaluation

  assert( v_i->data != NULL);  // v_i is the output node
  assert( k_idx < v_i->inputs->cnt);
  assert( self->n_args_compute == 1);

  const Value out_grad = (Value) v_i->adjoint;
  assert( out_grad != NULL);
  const Value v_k = get_val( v_i->inputs, 0);

  uint32_t out_shape_len = v_i->data->rank;
  uint32_t *out_shape = v_i->data->shape;

  uint32_t in_shape_len = v_k->data->rank;
  uint32_t *in_shape = v_k->data->shape;

  if ( out_shape_len == 0 || out_shape_len == 1) {
    return v_summation( out_grad, 0, NULL, FALSE);
  }
  if ( out_shape_len != in_shape_len) {
    fprintf( stderr, "broadcast_gradient: mismatching shapes in, out\n");
    assert( FALSE);
    return NULL;
  }
  assert( out_shape_len == in_shape_len);

  const uint16_t axes_len = in_shape_len;
  uint8_t *axes = alloca( axes_len * sizeof( uint8_t));
  uint16_t axes_cnt = 0;
  memset( axes, 0, sizeof( uint8_t) * axes_len);
  assert( in_shape_len < 255);

  // set the axes we use to 1, unlike numpy...
  axes_cnt = 0;
  for ( uint16_t i = 0; i < in_shape_len; i++) {
    const uint32_t s_in = in_shape[i];
    const uint32_t s_out = out_shape[i];
    if ( s_in == 1) {
      axes[axes_cnt++] = 1;
    }
  }

  // we use only the axes we set...
  assert( axes_cnt < axes_len);

  // keep_dims == FALSE
  Value v1 = v_summation( out_grad, axes_cnt, axes, FALSE);
  return v_reshape( v1, in_shape_len, in_shape);
}

Op BroadcastTo_new( const uint16_t rank, const t_shape shape) {
  Op op = op_new( broadcast_tensor, broadcast_gradient, 1);
  op->sub_type = AG_BROADCAST_TO;
  op->u.shape.rank = rank;
  op->u.shape.shape = alloc_shape( rank, shape);
  return op;
}

Value v_broadcast( Value v, const uint16_t rank, const t_shape shape) {
  Value vv = value_new();
  vv->op = BroadcastTo_new( rank, shape);
  vv->inputs = alloc_inputs( 1, v);
  vv->data = (vv->op->compute)(vv->op, 1, vv->inputs);
  return vv;
}

// ================== MatMul ========================

static t_tensor matmul_tensor( const void *_self,
			       const uint32_t n_args,
			       const l_list args) {
  const Op self = (const Op) _self;
  assert( n_args == 2);
  assert( self->sub_type == AG_MATMUL);

  const t_tensor t1 = get_arg( args, 0);
  const t_tensor t2 = get_arg( args, 1);

  const t_tensor t3 = t_matmul( t1, t2, NULL);

  return t3;
}

static void *matmul_gradient(const void *_self,
			     const uint32_t k_idx,
			     const void *_v_i) {
  const Op self = (const Op) _self;

  const Value v_i = (Value) _v_i; // node i in forward evaluation
  assert( v_i->adjoint != NULL);

  assert( k_idx < v_i->inputs->cnt);
  assert( self->n_args_compute == 2);

  Value v_bar_k_i = NULL;
  // v = a * b, gradient: matmul( out_grad, transpose(b)), matmul(transpose(a), out_grad)
  if ( k_idx == 0) {
    const Value b = get_val( v_i->inputs, 1);
    const Value b_transpose = v_transpose( b, 0, NULL);

    assert( b != NULL);
    assert( b_transpose != NULL);

    v_bar_k_i = v_matmul( v_i->adjoint, b_transpose);    

  } else {
    assert( k_idx == 1);
    const Value a = get_val( v_i->inputs, 0);
    const Value a_transpose = v_transpose( a, 0, NULL);

    assert( a != NULL);
    assert( a_transpose != NULL);

    v_bar_k_i = v_matmul( a_transpose, v_i->adjoint);

  }
  
  return v_bar_k_i;
}

Op MatMul_new() {
  Op op = op_new( matmul_tensor, matmul_gradient, 2);
  op->sub_type = AG_MATMUL;
  return op;
}

Value v_matmul( const Value v1, const Value v2) {
  Value v = value_new();
  v->op = MatMul_new();
  v->inputs = alloc_inputs( 2, v1, v2);
  v->data = (v->op->compute) (v->op, 2, v->inputs);
  return v;
}

// ============================ Negate ===================

static t_tensor negate_tensor( const void *_self,
			       const uint32_t n_args,
			       const l_list args) {
  const Op self = (const Op) _self;
  assert( n_args == 1);
  assert( self->sub_type == AG_NEGATE);

  const t_tensor t1 = get_arg( args, 0);
  return t_negate( t1, NULL);
}

static void *negate_gradient( const void *_self,
			      const uint32_t k_idx,
			      const void *_v_i) {
  const Op self = (const Op) _self;

  const Value v_i = (Value) _v_i; // node i in forward evaluation
  assert( k_idx == 0);
  assert( v_i->inputs->cnt == 1);

  // v = -x, dv/dx = -1 -> v_bar_k_i = v_bar_i * -1
  Value v_bar_k_i = v_negate( v_i->adjoint);

  return v_bar_k_i;
}

Op Negate_new() {
  Op op = op_new( negate_tensor, negate_gradient, 1);
  op->sub_type = AG_NEGATE;
  return op;
}

Value v_negate( const Value u) {
  Value v = value_new();
  v->op = Negate_new();
  v->inputs = alloc_inputs( 1, u);
  v->data = (v->op->compute)(v->op, 1, v->inputs);
  return v;
}

//============================= Log (ln) ==================================

static void *log_gradient(const void *_self,
			  const uint32_t k_idx,
			  const void *_v_i) {
  const Op self = (const Op) _self;

  const Value v_i = (Value) _v_i; // node i in forward evaluation
  assert( k_idx == 0);
  assert( v_i->inputs->cnt == 1);

  const Value v_k = get_val( v_i->inputs, 0);
  // v = ln(x), dv/dx = 1/x -> v_bar_k_i = v_bar_i / v_k
  Value v_bar_k_i = v_div( v_i->adjoint, v_k);

  return v_bar_k_i;
  
}

static t_tensor log_tensor( const void *_self,
			    const uint32_t n_args,
			    const l_list args) {
  const Op self = (const Op) _self;
  assert( n_args == 1);
  assert( self->sub_type == AG_LOG);

  const t_tensor t1 = get_arg( args, 0);
  return t_log( t1, NULL);
}

Op Log_new() {
  Op op = op_new( log_tensor, log_gradient, 1);
  op->sub_type = AG_LOG;
  return op;
}

Value v_log( const Value u) {
  Value v = value_new();
  v->op = Log_new();
  v->inputs = alloc_inputs( 1, u);
  v->data = (v->op->compute)(v->op, 1, v->inputs);
  return v;
}


// ==================================== Exp ==========================

static t_tensor exp_tensor( const void *_self,
			    const uint32_t n_args,
			    const l_list args) {
  const Op self = (const Op) _self;
  assert( n_args == 1);
  assert( self->sub_type == AG_EXP);

  const t_tensor t1 = get_arg( args, 0);
  // e^x
  return t_exp( t1, NULL);
}

static void *exp_gradient( const void *_self,
			   const uint32_t k_idx,
			   const void *_v_i) {
  const Op self = (const Op) _self;

  assert( k_idx == 0);
  assert( self->n_args_compute == 1);
  assert( self->sub_type == AG_EXP);
  
  const Value v_i = (Value) _v_i; // node i in forward evaluation

  // the partial derivative of e^x is is e^x... i.e. v_i
  assert( v_i->data != NULL);

  const Value exp_a = v_i; // v_i is the output of the exponentiation e^v_k
  // we create a new node computing the partial adjoint 
  Value v_bar_k_i = v_mul( v_i->adjoint, exp_a);
  return v_bar_k_i;
}

Op Exp_new() {
  Op op = op_new( exp_tensor, exp_gradient, 1);
  op->sub_type = AG_EXP;
  return op;
}

Value v_exp( const Value u) {
  Value v = value_new();
  v->op = Exp_new();
  v->inputs = alloc_inputs( 1, u);
  v->data = (v->op->compute)(v->op, 1, v->inputs);
  return v;
}

// ============================ ReLU =============================

static t_tensor relu_tensor( const void *_self,
			     const uint32_t n_args,
			     const l_list args) {
  const Op self = (const Op) _self;
  assert( n_args == 1);
  assert( self->sub_type == AG_RELU);

  const t_tensor t1 = get_arg( args, 0);
  assert( t_assert( t1));
  
  return t_relu( t1, NULL);
}

static void *relu_gradient( const void *_self,
			    const uint32_t k_idx,
			    const void *_v_i) {
  const Op self = (const Op) _self;
  assert( self->n_args_compute == 1);

  const Value v_i = (Value) _v_i; // node i in forward evaluation
  const Value v_k = get_val( v_i->inputs, 0);
  
  const Value dv_i_dv_k = v_relu_deriv( v_k);
  Value v_bar_k_i = v_mul( v_i->adjoint, dv_i_dv_k);

  return v_bar_k_i;
}

Op ReLU_new() {
  Op op = op_new( relu_tensor, relu_gradient, 1);
  op->sub_type = AG_RELU;
  return op;
}

Value v_relu( const Value u) {
  Value v = value_new();
  v->op = ReLU_new();
  v->inputs = alloc_inputs( 1, u);
  v->data = (v->op->compute)(v->op, 1, v->inputs);
  return v;
}

// =============================== Sign ========================================

static t_tensor sign_tensor( const void *_self,
			    const uint32_t n_args,
			    const l_list args) {
  const Op self = (const Op) _self;
  assert( n_args == 1);
  assert( self->sub_type == AG_EXP);

  const t_tensor t1 = get_arg( args, 0);
  return t_sign( t1, NULL);
}

Op Sign_new() {
  Op op = op_new( sign_tensor, NULL, 1);
  op->sub_type = AG_SIGN;
  return op;
}

Value v_sign( const Value u) {
  Value v = value_new();
  v->op = Sign_new();
  v->inputs = alloc_inputs( 1, u);
  v->data = (v->op->compute)(v->op, 1, v->inputs);
  return v;
}


// ================================= Relu Deriv ================================

static t_tensor relu_deriv( const void *_self,
			    const uint32_t n_args,
			    const l_list args) {
  const Op self = (const Op) _self;
  assert( n_args == 1);
  assert( self->sub_type == AG_RELU_DERIV);

  const t_tensor t1 = get_arg( args, 0);
  return t_relu_deriv( t1, NULL);
}

Op ReluDeriv_new() {
  // we leave the gradient() function NULL as it's never needed
  Op op = op_new( relu_deriv, NULL, 1);
  op->sub_type = AG_RELU_DERIV;
  return op;
}

// for consistency reasons, the relu_deriv is a Value
Value v_relu_deriv( const Value u) {
  Value v = value_new();
  v->op = ReluDeriv_new();
  v->inputs = alloc_inputs( 1, u);
  v->data = (v->op->compute)(v->op, 1, v->inputs);
  return v;
}

// ================================ log soft max =================================
static t_tensor log_soft_max( const void *_self,
			      const uint32_t n_args,
			      const l_list args) {
  const Op self = (const Op) _self;
  assert( n_args == 1);
  assert( self->sub_type == AG_LOG_SOFT_MAX);

  const t_tensor z = get_arg( args, 0);
  assert( t_assert( z));

  t_tensor t = t_log_softmax( z, NULL);

  return t;
}

static void *log_soft_max_gradient( const void *_self,
				    const uint32_t k_idx,
				    const void *_v_i) {
  const Op self = (const Op) _self;
  assert( self->sub_type == AG_LOG_SOFT_MAX);

  // the output of log_soft_max is log-soft-max...
  const Value lsm_z = (Value) _v_i; // output node in fwd evaluation

  const Value out_grad = (Value) lsm_z->adjoint;
  assert( out_grad != NULL);

  /*
    sm_z = exp( lsm_z) ## softmax(z)
    sum_grad = summation(out_grad, axes=1)  # Sum along softmax axis
    print( f'sum_grad = {sum_grad}')
    print( f'sm_z = {sm_z}')
    ## the sum_grad term was suggested by chatGPT...
    ##  out_grad * I - sum_grad * softmax(z)
    return out_grad - sm_z * sum_grad  # Apply (I - softmax) * grad_output
  */

  Value sm_z = v_exp( lsm_z);
  const uint8_t axes[] = {0,1};  // along rows... axes==1
  Value sum_grad = v_summation( out_grad, 2, axes, TRUE);
  
  Value sm_z_times_sum_grad = v_mul( sm_z, sum_grad);
  
  // Value neg_sm_z_times_sum_grad = v_negate( sm_z_times_sum_grad);

  // out_grad - sm_z * sum_grad
  Value grad = v_sub( out_grad, sm_z_times_sum_grad);
  
  return grad;
}

Op LogSoftMax_new() {
  Op op = op_new( log_soft_max, log_soft_max_gradient, 1);
  op->sub_type = AG_LOG_SOFT_MAX;
  return op;
}

Value v_log_soft_max(const Value Z) {
  Value v = value_new();
  v->op = LogSoftMax_new();
  v->inputs = alloc_inputs( 1, Z);
  v->data = (v->op->compute) (v->op, 1, v->inputs);
  return v;
}

// =================================  log sum exp ===============================

static t_tensor log_sum_exp( const void *_self,
			      const uint32_t n_args,
			      const l_list args) {
  const Op self = (const Op) _self;
  assert( n_args == 1);
  assert( self->sub_type == AG_LOG_SUM_EXP);

  const t_tensor z = get_arg( args, 0);
  assert( t_assert( z));
  assert( t_is2D( z));
  
  t_tensor t = t_log_sumexp( z, NULL);

  return t;
}

static void *log_sum_exp_gradient( const void *_self,
				   const uint32_t k_idx,
				   const void *_v_i) {
  const Op self = (const Op) _self;
  assert( self->sub_type == AG_LOG_SUM_EXP);

  // the output of log_sum_exp is log-sum-exp...
  const Value lse_z = (Value) _v_i; // output node in fwd evaluation
  const Value z = get_val( lse_z->inputs, 0);

  const Value out_grad = (Value) lse_z->adjoint;
  assert( out_grad != NULL);

  /*
    ## ChatGPT can show the gradient of grad_Z (LSE(Z)) = e^(Z - LSE(Z))...
    Z = node.inputs[0] ## self.Z
    if self.axes:
      shape = [1] * len(Z.shape)
      j = 0
      for i in range(len(shape)):
        if i not in self.axes:
          shape[i] = node.shape[j]
          j += 1
      node_new = node.reshape(shape)
      grad_new = out_grad.reshape(shape)
    else:
      node_new = node
      grad_new = out_grad
    return grad_new * exp(Z - node_new)
   */

  const Value Z = get_val( lse_z->inputs, 0);

  Value node_new = NULL;
  Value grad_new = NULL;
  if ( self->u.axes.axes_len > 0) {
    // reconstitute the original shape...
    const uint32_t rank_z = t_rank( z->data);
    uint32_t shape[ rank_z];

    // we reshape things to be 2D...
    uint32_t j = 0;
    for ( uint32_t i = 0; i < rank_z; i++) {
      assert( i < self->u.axes.axes_len);
      if ( self->u.axes.axes[i] == 0) {
	assert( j < lse_z->data->rank);
	shape[i] = lse_z->data->shape[j++];
      } else if ( self->u.axes.axes[i]== 1) {
	shape[i] = 1;
      } else {
	assert( 0);
      }
    }
  
    node_new = v_reshape( lse_z, rank_z, shape);
    grad_new = v_reshape( out_grad, rank_z, shape);
    
  } else {
    node_new = lse_z;
    grad_new = out_grad;
  }
  
  // const Value neg_lse_z = v_negate( node_new);
  const Value z_minus_lse_z = v_sub( Z, lse_z);
  const Value exp_z = v_exp( z_minus_lse_z);
  const Value grad = v_mul( grad_new, exp_z);

  return grad;
}

Op LogSumExp_new( const uint16_t axes_len, const uint8_t axes[]) {
  Op op = op_new( log_sum_exp, log_sum_exp_gradient, 1);
  op->sub_type = AG_LOG_SUM_EXP;
  if ( axes_len > 0) {
    assert( axes != NULL);
    op->u.axes.axes_len = axes_len;
    op->u.axes.axes = alloc_axes( axes_len, axes);
  } else {
    assert( axes == NULL);
    op->u.axes.axes_len = 0;
    op->u.axes.axes = NULL;
  }
  return op;
}

Value v_log_sum_exp( const Value Z, const uint16_t rank, const uint8_t axes[]) {

  assert(( rank != 0 && axes != NULL) || (rank == 0 && axes == NULL));

  Value v = value_new();
  v->op = LogSumExp_new( rank, axes);
  v->inputs = alloc_inputs( 1, Z);
  v->data = (v->op->compute) (v->op, 1, v->inputs);
  return v;
}

// =================================  end of ops ================================

#if 0
t_tensor ag_add( const t_tensor a, const t_tensor b) {
  Op op = EWiseAdd_new();
  const t_tensor t = (*op->compute)(op, 2, a, b);
  op_free( op);
  return t;
}

t_tensor ag_add_scalar( const t_tensor a, const double s) {
  Op op = AddScalar_new( s);
  const t_tensor t = (*op->compute)(op, 1, a);
  op_free( op);
  return t;
}

t_tensor ag_reshape( const t_tensor t, const uint16_t rank,
		     const t_shape shape) {
  Op op = Reshape_new( rank, shape);
  const t_tensor tt = (*op->compute) (op, 1, t);
  op_free( op);
  return tt;
}

t_tensor ag_negate( const t_tensor t) {
  Op op = Negate_new();
  const t_tensor tt = (*op->compute) (op, 1, t);
  op_free( op);
  return tt;
}
#endif

// frees a value, including its op and data tensor...
void v_free( void *_v) {
  Value v = (Value) _v;
  op_free( v->op);
  l_free( v->inputs);
  // sometimes we want to keep the tensor...
  if ( !v->shared_data)
    t_free( v->data);
  l_free( v->node_to_grad);
  v->adjoint = NULL;
  MEM_FREE( _v);
}

uint8_t v_is_tensor( const Value v) {
  assert( ag_is_value( v));
  return v->op == NULL;
}

Value v_tensor_to_value( const t_tensor t, const uint8_t shared_data) {
  Value v = value_new();
  v->data = t;
  v->shared_data = shared_data;
  return v;
}


// creates a variable for a given tensor, wrapping a tensor into a value
Value v_tensor( const t_tensor t) {
  Value v = value_new();
  v->data = t;
  v->shared_data = FALSE;
  return v;
}

Value v_randb( const uint16_t rank, const t_shape shape,
	       const float p) {
  Value v = value_new();
  v->data = t_randb( p, rank, shape, T_FLOAT);
  return v;
}

Value v_ones( const uint16_t rank, const t_shape shape) {
  Value v = value_new();
  v->data = t_new_tensor( rank, shape, T_FLOAT, NULL);
  t_fill( v->data, 1.0);
  return v;
}

Value v_zeros( const uint16_t rank, const t_shape shape) {
  Value v = value_new();
  v->data = t_new_tensor( rank, shape, T_FLOAT, NULL);
  return v;
}

Value v_minus_ones( const uint16_t rank, const t_shape shape) {
  Value v = value_new();
  v->data = t_new_tensor( rank, shape, T_FLOAT, NULL);
  t_fill( v->data, -1.0);
  return v;
}


void v_dump( const Value v, const uint8_t verbose) {
  fprintf( stderr, "node: 0x%p: ", (void *) v);
  if ( v->op == NULL) {
    fprintf( stderr, "no-op ");
  } else {
    fprintf( stderr, "op: %d ", v->op->sub_type);
  }

  fprintf( stderr, "data: ");
  if ( v->data == NULL)
    fprintf( stderr, "None\n");
  else {
    fprintf( stderr, "\n");
    t_dump( v->data, verbose, FALSE);
  }
  
  fprintf( stderr, "inputs: ");
  if ( v->inputs == NULL)
    fprintf( stderr, "NULL\n");
  else
    l_dump( v->inputs);

  fprintf( stderr, "adjoint: ");
  if ( v->adjoint == NULL)
    fprintf( stderr, "NULL\n");
  else
    fprintf( stderr, "0x%p\n", v->adjoint);

  fprintf( stderr, "node_to_grad: ");
  if ( v->node_to_grad == NULL)
    fprintf( stderr, "NULL\n");
  else
    l_dump( v->node_to_grad);

  fprintf( stderr, "\n");
}

// ===================================================================================================


// convenience to append a value to the list
static void l_append_value( const l_list l,
			    const Value v,
			    const uint8_t unique) {
  l_el el;
  el.ptr = (void *) v;
  if ( unique)
    l_append_unique( l, el);
  else
    l_append( l, el);
}


#if 0
/**
   starting with out nodes add nodes into a list by
   recursively going backwards using a node's input list and
   ignoring duplicates

   nodes: total list of nodes
   cur_nodes: a list of nodes currently visited, possibly added to nodes

*/
static void ag_get_nodes( const l_list nodes, const l_list cur_nodes) {
  for ( unsigned int i = 0; i < cur_nodes->cnt; i++) {
    l_el el = l_get( cur_nodes, i);
    l_append_unique( nodes, el);

    // recurse on v's inputs.... until we reach a root node
    Value v = (Value) el.ptr;
    if ( v->inputs != NULL && v->inputs->cnt > 0)
      ag_get_nodes( nodes, v->inputs);
    // else no inputs -> root node -> done
  }
}

// given a set of output nodes (i.e. nodes which are not input to
// any other node) construct the list of all nodes
l_list ag_get_fwd_nodes( const uint32_t nbr_nodes,
		      ...) {
  l_list out_nodes = l_new( nbr_nodes, T_PTR, NULL);

  va_list argptr;
  va_start( argptr, nbr_nodes);
  
  for ( unsigned int i = 0; i < nbr_nodes; i++) {
    const Value v = va_arg( argptr, Value);
    l_append_value( out_nodes, v, TRUE);
  }
  va_end( argptr);

  l_list fwd_nodes = l_new( nbr_nodes, T_PTR, v_free);
  ag_get_nodes( fwd_nodes, out_nodes);
  return fwd_nodes;
}
#endif

static void visit_node( const l_list sl,
			const Value v) {
  if ( v->visited == PERM_VISITED)
    return;
  if ( v->visited == TEMP_VISITED) {
    fprintf( stderr, "cycle in DAG?");
    exit( -1);
  }
  v->visited = TEMP_VISITED;

  if ( v->inputs != NULL)  {
    for ( int i = 0; i < v->inputs->cnt; i++) {
      Value vv = (Value) ((l_get( v->inputs, i).ptr));
      visit_node( sl, vv);
    }
  }

  v->visited = PERM_VISITED;
  l_append_value( sl, v, TRUE);
}

// using the output node as root, do a recursive depth-first post-order
// see https://en.wikipedia.org/wiki/Topological_sorting
l_list ag_get_topo_sorted_nodes( const Value root) {

  // first we unmark all the fwd nodes we ever created.
  for ( int i = 0 ; i < fwd_val_list->cnt; i++) {
    const Value v = (Value) (l_get( fwd_val_list, i).ptr);
    v->visited = NOT_VISITED;
  }

  // do not free elements when freeing list...
  // the size is including values which are in fact not part of the computational
  // graph. for example the gradient of the output_tensor is a value but not part
  // of the input->output dependencies of values
  l_list sl = l_new( fwd_val_list->cnt, T_PTR, NULL);

  // start recursion at root
  visit_node( sl, root);

  // reverse sl... since we appended and did not prepend...
  l_list rsl = l_reverse( sl);
  l_free( sl);
  return rsl;
}


static void append_to_node_to_grad( const Value n, const Value m) {
  // alloc node_to_grad if needed
  if ( n->node_to_grad == NULL)
    n->node_to_grad = l_new( 2, T_PTR, NULL);
  
  // and insert m into n->node_to_grad
  l_append_value( n->node_to_grad, m, FALSE);
}

static Value compute_adjoint( const l_list node_to_grad) {
  assert( node_to_grad != NULL && node_to_grad->cnt > 0);
  
  if ( node_to_grad->cnt == 1) {
    const Value v = get_val( node_to_grad, 0);
    return v;
  } else {
    // else compute the sum of the values in the list
    Value adj = value_new();
    adj->inputs = l_new( node_to_grad->cnt, T_PTR, NULL);

    // iterate over list and accumulate sum
    Value v = get_val( node_to_grad, 0);
    t_tensor grad = t_clone( v->data);
    for ( uint32_t idx = 1; idx < node_to_grad->cnt; idx++) {
      v = get_val( node_to_grad, idx);
      grad = t_add( grad, v->data, grad);
      // we keep track of the input nodes
      l_append_value( adj->inputs, v, FALSE);
    }
    // set the value of the gradient to the sum of partial ajdoints
    adj->data = grad;
    return adj;
  }
  return NULL;
}


// given the output node k and input node i (of k) compute v_bar_k_i
// using v_bar_i (out_gradient) and the derivative d(v_i)/d(v_k) based on output node i
static Value compute_partial_adjoint( const uint32_t k_idx,
				      const Value node_i) {

  assert( node_i != NULL);
  assert( node_i->op != NULL);
  assert( node_i->op->gradient != NULL);
  assert( node_i->adjoint != NULL);
  assert( k_idx < node_i->inputs->cnt);

  Value v =  
    (node_i->op->gradient) (node_i->op, k_idx, node_i);

  return v;
}

// creates a value of all ones using dimensions of argument v
static Value v_all_ones( const Value v) {
  const t_tensor v_data = v->data;
  const t_tensor ones = t_ones( v_data->rank, v_data->shape, v_data->dtype);
  return v_tensor_to_value( ones, FALSE);
}

// see autograd.py:backward()
// out_grad == NULL => allocate a tensor of all ones...: convenience
void ag_gradient( const Value output_tensor, Value out_grad) {

  // set bwd motion flag and reset bwd graph
  ag_set_mode( AG_BWD_MODE);
  l_reset( bwd_val_list); // frees contained elements

  if ( out_grad == NULL) {
    out_grad = v_all_ones( output_tensor);
    output_tensor->adjoint = out_grad;
  }
  

  assert( output_tensor != NULL);
  assert( out_grad != NULL);
  assert( output_tensor->adjoint == out_grad);
  
  // compute the reverse topo sorted nodes, starting with output_tensor

#if 0
  if ( debug)
    ag_dump( bwd_val_list, FALSE);
#endif
  
  l_list sorted_nodes = ag_get_topo_sorted_nodes( output_tensor);

#if 0
  if ( debug)
    ag_dump( sorted_nodes, FALSE);
#endif
  
  // since for the output_tensor we have adjoint == out_grad we don't need to
  // do this here. appending the out_grad to node_to_grad[output_tensor], as singleton
  // and summing the singleton list would re-compute the gradient and in the C world
  // cause all sorts of memory mgmt issues....
  // set node_to_grad for out
  // append_to_node_to_grad( output_tensor, out_grad);

  for ( int i = 0; i < sorted_nodes->cnt; i++) {
    
    const Value node_i = get_val( sorted_nodes, i);

    // fprintf( stderr, "node %p\n", node_i);
    
    // special case for the output node for which we already have the adjoint (gradient) computed
    if ( node_i == output_tensor) {
      assert( node_i->adjoint == out_grad);
    } else {
      node_i->adjoint = compute_adjoint( node_i->node_to_grad);
    }
    
    // tensor as Value node... 
    if ( node_i->op == NULL)
      continue;

    const l_list inputs = node_i->inputs;
    if ( inputs != NULL && inputs->cnt > 0) {
      for ( int k = 0; k < inputs->cnt; k++) {
	const Value node_k = get_val( inputs, k);

	const Value v_bar_k_i = compute_partial_adjoint( k, node_i);
	append_to_node_to_grad( node_k, v_bar_k_i);
      }
    }
  }
  
  // elements of list *not* freed...
  l_free( sorted_nodes);
  // l_free( nodes);

  // stop the backward motion...
  ag_set_mode( AG_FWD_MODE);
  
}

void ag_dump( l_list nodes, const uint8_t verbose) {
  for ( int i = 0; i < nodes->cnt; i++) {
    const Value v = get_val( nodes, i);
    v_dump( v, verbose);
  }
}

static unsigned int get_y_max( const t_tensor y) {
  t_tensor y_max = t_max( y, NULL, FALSE);
  const unsigned int m = (unsigned int) t_scalar( y_max);
  t_free( y_max);
  return m;
}

// from 2D into 1D
static t_tensor flatten_images( t_tensor t) {
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

static void set_slice_idx( uint32_t s[][2], const unsigned int idx,
			   const unsigned int lb, const unsigned int ub) {
  s[idx][0] = lb;
  s[idx][1] = ub;
}

static int32_t find_idx_of_one( const t_tensor y, const uint32_t row_idx) {
  assert( y->rank == 2);
  const uint32_t nbr_cols = y->shape[1];
  for ( uint32_t j = 0; j < nbr_cols; j++) {
    const double d = t_2D_get( y, row_idx, j);
    if ( d >= 0.95 && d <= 1.05)
      return j;
  }
  return -1;
}


/*
  to painfully compute the gradient of Z.
  which is done row-wise, traversing the columns
  and we can't use python numpy's vector features
*/
static t_tensor comp_grad_Z( const t_tensor Z_grad,
			 const t_tensor Z_exp,
			 const t_tensor sum_rows_Z,
			 const t_tensor y,
			 const uint32_t row) {
  assert( Z_grad->rank == 2 && Z_exp->rank == 2 && sum_rows_Z->rank == 1 && y->rank == 1);
  assert( t_same_shape( Z_grad, Z_exp));
  assert( t_same_shape( sum_rows_Z, y));
  assert( row < y->shape[0]);

  /*
    ## based on chatGPT... y_grad stays zero
    ## dL/dZ = sigma - y 
    Z_grad[i] = Z_exp[i]/sum_rows_Z[i]
    ## y_grad[i] = - np.log( Z_grad[i])
    Z_grad[i] = Z_grad[i] - y[i]
  */
  const uint32_t n_cols = Z_grad->shape[1];

  // get the value of the summed rows of Z
  const double sum_rows_Z_i = t_1D_get( sum_rows_Z, row);
  
  // label value, indicates in which col we subtract 1 or 0...
  // y is a vector of labels, not a one-hot vector matrix
  const uint32_t y_i = t_1D_get( y, row);
  
  for ( uint32_t j = 0; j < n_cols; j++) {
    const double z_exp_i = t_2D_get( Z_exp, row, j);
    double zg = z_exp_i / sum_rows_Z_i;

    if ( j == y_i) { // y_j == 1
      zg -= 1.0;
    } // else the one-hot vector would be 0...

    t_2D_set( Z_grad, row, j, zg);
  }

  // returns a t_tensor
  return Z_grad;
}

/**
   Return softmax loss.  Note that for the purposes of this assignment,
   you don't need to worry about "nicely" scaling the numerical properties
   of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 1D Tensor of shape (batch_size)
            containing true class-labels. *NOT* a matrix of one-hot vectors...

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
*/
static double softmax_loss( const Value Z, const t_tensor y) {

  /*
    the formulas for the gradients of Z and y are based on chatGPt
    for y, the gradient is all zeros
    for Z, we have dL/dZ = sigma - y and, since we are batching, we
    must use the average here...
  */
  const uint8_t Z_type = DTYPE( Z->data);
  const uint32_t *Z_shape = SHAPE( Z->data);

  const uint32_t y_len = y->shape[0];

  // we build the gradient for Z...
  const t_tensor Z_grad = t_new( 2, Z_shape, Z_type);

  // use the tensor API for our calculations...
  const t_tensor Z_exp = t_exp( Z->data, NULL);

  // compute the sum along the rows of Z_exp
  const uint8_t axes[] = {0,1};
  t_tensor sum_rows_Z = t_sum( Z_exp, axes, FALSE);

  // expect a 1 D tensor of the length of y == nbr of rows...
  assert( sum_rows_Z->rank == 1);
  assert( t_1D_len( sum_rows_Z) == y_len);

  const t_tensor log_sum_rows_Z = t_log( sum_rows_Z, NULL);

  // 1 D tensor, i.e. vector to hold the losses for each sample
  const t_tensor res = t_new_vector( Z_shape[0], Z_type, NULL);
  
  for ( uint32_t i = 0; i < y_len; i++) {
    /*
      ## y[i] here is a one-hot-vector of appropriate length
      ## we need to find the index of which the element is close to 1
      y_idx = find_idx_of_one( y[i])
      res[i] -= Z[i, y_idx]

      ## based on chatGPT... y_grad stays zero
      ## dL/dZ = sigma - y 
      Z_grad[i] = Z_exp[i]/sum_rows_Z[i]
      ## y_grad[i] = - np.log( Z_grad[i])
      Z_grad[i] = Z_grad[i] - y[i]
    */

    // get the label for row i. since we don't have a one-hot vector matrix, we just get
    // the label value...
    int32_t y_idx = (int32_t) t_1D_get( y, i);

    // loss for i-th sample: a scalar...
    // loss(z, y) = log (sum( exp z_i)) - z_y
    
    const double z_y = t_2D_get( Z->data, i, y_idx);

    // res[i] == log (sum_k exp Z_k[i])...
    const double lsrz = t_1D_get( log_sum_rows_Z, i);

    const double loss_i = lsrz - z_y;
    t_1D_set( res, i, loss_i);

    // handle the gradient of Z
    // we miss the elegance of python numpy here...
    comp_grad_Z( Z_grad, Z_exp, sum_rows_Z, y, i);

  }

  /*
    ## set the gradients of the input args
    ## for Z, compute the average of the gradients...
    Z_t.grad = ndl.Tensor( Z_grad/len(y))
  */
  t_div_scalar( Z_grad, (double) y_len); // in-situ
  Z->adjoint = v_tensor_to_value( Z_grad, FALSE);
  
  /*
    compute the average loss...
    ## len(y) == batch_size
    res = np.sum( res)/len( y)
    this is a scalar...!
  */
  const t_tensor sum_res = t_sum( res, NULL, FALSE);
  assert( sum_res->rank == 0); // expect a 0D tensor...

  double avg_loss = t_scalar( sum_res)/((double) y_len);

  t_free( sum_res);
  t_free( res);
  t_free( log_sum_rows_Z);
  t_free( sum_rows_Z);
  t_free( Z_exp);

  return avg_loss;
}

static void nn_batch( const t_tensor X_t,  // 2D tensor
		      const t_tensor y_t,  // 1D tensor of y-labels
		      const t_tensor W_1_t,
		      const t_tensor W_2_t,
		      const double lr) {

  ag_set_mode( AG_FWD_MODE); // record all the values in forward mode

  const uint32_t num_examples = SHAPE_DIM( X_t, 0);    // nbr of rows in batch
  const uint32_t input_dim    = SHAPE_DIM( X_t, 1);    // size of input vectors
  const uint32_t hidden_dim   = SHAPE_DIM( W_1_t, 1);   // size of hidden layer
  const uint32_t num_classes  = SHAPE_DIM( W_2_t, 1);   // size of output

  // assert( X_shape.shape[0] == y_shape.shape[0]);  
  assert( SHAPE_DIM( X_t, 0) == SHAPE_DIM( y_t, 0));     // nbr of rows, input & output
  assert( SHAPE_DIM( X_t, 1) == SHAPE_DIM( W_1_t, 0));   // nbr of cols input == nbr of rows W1
  assert( SHAPE_DIM( W_1_t, 1) == SHAPE_DIM( W_2_t, 0)); // nbr of cols W1 == nbr of rows W2

#if 0
  fprintf( stderr, "num_examples: %d input_dim: %d, hidden_dim: %d, num_classes: %d\n",
	   num_examples, input_dim, hidden_dim, num_classes);
#endif

  // from a 1D vector of labels to a 2D one-hot matrix
  // t_tensor y_h = t_one_hot( y_t);
  
  // create values from tensor for X_t. we keep y_t as 1D tensor of y-labels...
  Value X = v_tensor_to_value( X_t, TRUE); // shared tensor...

  assert( X != NULL);
  assert( y_t != NULL);

  // we share the tensors... just wrapping the Value around
  Value W_1 = v_tensor_to_value( W_1_t, TRUE);
  Value W_2 = v_tensor_to_value( W_2_t, TRUE);
  
  assert( W_1 != NULL);
  assert( W_2 != NULL);

  Value X_W1 = v_matmul( X, W_1);
  assert( X_W1 != NULL);
  
  Value Z_1 = v_relu( X_W1);
  assert( Z_1 != NULL);  
  
  assert( V_SHAPE( Z_1, 0) == num_examples);
  assert( V_SHAPE( Z_1, 1) == hidden_dim);
  
  Value Z_1_W_2 = v_matmul( Z_1, W_2);

  double avg_loss = softmax_loss( Z_1_W_2, y_t);
  assert( Z_1_W_2->adjoint != NULL);

  /*
    ## get the gradients...
    Z_1_W_2.backward( Z_1_W_2.grad)
  */

  // this implicitly switches the value mode to BWD
  ag_gradient( Z_1_W_2, Z_1_W_2->adjoint);
  
  /*
    ## print( f"Z_1_W_2.grad = {Z_1_W_2.grad}")

    grad_W_1 = W_1.grad
    grad_W_2 = W_2.grad

    grad_W_1 = ndl.mul_scalar( grad_W_1, -lr)
    grad_W_2 = ndl.mul_scalar( grad_W_2, -lr)

    W_1 = ndl.add( W_1, grad_W_1)
    W_2 = ndl.add( W_2, grad_W_2)
    
    return W_1.detach(), W_2.detach()

  */
  const t_tensor grad_W_1 = ((Value) W_1->adjoint)->data;
  const t_tensor grad_W_2 = ((Value) W_2->adjoint)->data;
  assert( grad_W_1 != NULL);
  assert( grad_W_2 != NULL);

  t_mul_scalar( grad_W_1, -lr);
  t_mul_scalar( grad_W_2, -lr);

  t_add( W_1_t, grad_W_1, W_1_t); // in-situ
  t_add( W_2_t, grad_W_2, W_2_t); // in-situ
    
  // clean up the graph and its contained Values.
  l_reset( ag_get_val_list( AG_FWD_MODE));
  l_reset( ag_get_val_list( AG_BWD_MODE));

}

void ag_nn_epoch( const t_tensor X,
	       const t_tensor y,
	       const t_tensor W_1,
	       const t_tensor W_2,
	       const double lr,
	       const unsigned int batch_size,
	       const unsigned int hidden_dim) {

  ag_init();
  
  assert( t_rank( X) == 2);
  assert( t_rank( y) == 1);

  const unsigned int y_len = t_1D_len( y);
  const unsigned int nbr_batches = (unsigned int) ceil( y_len/batch_size);

  fprintf( stderr, "nbr samples: %d, nbr_batches: %d, batch_size; %d\n", y_len, nbr_batches, batch_size);

  // to slice the input data into batches...
  uint32_t X_i_idx[2][2];
  uint32_t Y_i_idx[1][2];

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

    // creates new tensors...
    t_tensor X_i = t_slice( X, X_i_idx, 2);  // X[lb:ub, :];
    t_tensor y_i = t_slice( y, Y_i_idx, 1);  // y[lb:ub];

    nn_batch( X_i, y_i, W_1, W_2, lr);

    mem_dump_tbl();

    t_free( X_i);
    t_free( y_i);
  }

}


Value v_xavier_uniform( const uint32_t fan_in, const uint32_t fan_out, const double gain);
Value v_xavier_normal( const uint32_t fan_in, const uint32_t fan_out, const double gain);

Value v_kaiming_uniform( const uint32_t fan_in, const uint32_t fan_out, const char *non_linearity) {
  t_tensor t = t_kaiming( KAIMING_UNIFORM, fan_in, fan_out, non_linearity, T_FLOAT);
  Value v = v_tensor( t);
  return v;
}

Value v_kaiming_normal( const uint32_t fan_in, const uint32_t fan_out, const char *non_linearity);

// given a 1 D array of y-labels generate a 2 D matrix ( y->shape[0], num_cols)
Value v_one_hot( const uint32_t num_cols, const Value y) {
  assert( v_is_tensor( y));
  assert( t_is1D( y->data));
  
  const uint32_t num_rows = y->data->shape[0];
  t_tensor t = t_one_hot( y->data, T_FLOAT, num_cols);
  return v_tensor( t);
}



#if 0
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

  t_free( mnist_images);
  t_free( mnist_labels);

  exit( 0);



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
  ag_dump( fwd_val_list, FALSE);

  ag_gradient( v4, NULL);

  fprintf( stderr, "\n\n");
  ag_dump( bwd_val_list, FALSE);

  l_free( fwd_val_list);
  l_free( bwd_val_list);
  
}
#endif
