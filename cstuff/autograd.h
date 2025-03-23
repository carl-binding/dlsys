#ifndef _AUTOGRAD_H_
#define _AUTOGRAD_H_

#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <stdint.h>

#include "list.h"
#include "tensor.h"

#define AG_EWISE_ADD 1
#define AG_ADD_SCALAR 2
#define AG_EWISE_MUL 3
#define AG_MUL_SCALAR 4
#define AG_POWER_SCALAR 5
#define AG_EWISE_DIV 6
#define AG_DIV_SCALAR 7
#define AG_TRANSPOSE 8
#define AG_SUMMATION 9
#define AG_RESHAPE 10
#define AG_BROADCAST_TO 11
#define AG_MATMUL 12
#define AG_NEGATE 13
#define AG_LOG 14
#define AG_EXP 15
#define AG_RELU 16
#define AG_EWISE_SUB 17
#define AG_EWISE_POW 18
#define AG_RELU_DERIV 19
#define AG_SIGN 20
#define AG_LOG_SOFT_MAX 21
#define AG_LOG_SUM_EXP 22

// unique type tags so we can check void ptrs...
#define AG_OP_TYPE_TAG    0x10101010
#define AG_VALUE_TYPE_TAG 0x20202020

typedef struct {
  uint32_t type_tag;
  uint8_t sub_type; 
  // computes the value from input args and op
  t_tensor (*compute) (const void *self, const uint32_t n_args, const l_list args);
  // self: Op,
  // k_idx: index of input arg to node i
  // node_i: node i in forward graph: output node
  // returns: Value
  // note: this differs from the python world where the gradient function
  // takes v_bar_i and v_i (from which the inputs are extracted and a list of
  // partial adjoints is returned, NOT a single Value...)
  // returns the partial adjoint = partial derivative against input * gradient of output node
  void * (*gradient) (const void *self,
		      const uint32_t k_idx, // into v_i->inputs
		      const void *v_i);

  uint16_t n_args_compute;
  union {
    // AG_ADD_SCALAR, AG_MUL_SCALAR, AG_DIV_SCALAR, AG_POWER_SCALAR
    t_tensor s; // scalar tensor, dtype T_DOUBLE

    // AG_SUMMATION, AG_LOG_SUM_EXP, AG_TRANSPOSE
    struct {  
      uint16_t axes_len;
      uint8_t *axes;
    } axes;

    // AG_RESHAPE, AG_BROADCAST_TO
    struct {
      uint16_t rank;
      uint32_t *shape;
    } shape;
  } u;
  uint8_t keep_dims; // AG_SUMMATION
} Op_struct, *Op;

uint8_t ag_is_op( const void *p);

Op EWiseAdd_new();
Op EWiseSub_new();
Op EWiseMul_new();
Op EWiseDiv_new();
Op EWisePow_new();

Op AddScalar_new( const double s);
#define SubScalar_new( s) AddScalar( -s)

Op MulScalar_new( const double s);
Op DivScalar_new( const double s);
Op PowerScalar_new( const double s);

Op Transpose_new( const uint16_t rank, const t_axes axes);
Op Summation_new(  const uint16_t rank, const t_axes axes, const uint8_t keep_dims);

Op Reshape_new( const uint16_t rank, const t_shape shape);
Op BroadcastTo_new( const uint16_t rank, const t_shape shape);

Op MatMul_new();
Op Negate_new();
Op Log_new();
Op Exp_new();
Op ReLU_new();

Op LogSoftMax_new();
Op LogSumExp_new(const uint16_t axes_len, const uint8_t axes[]);

#if 0
t_tensor ag_add( const t_tensor a, const t_tensor b);
t_tensor ag_add_scalar( const t_tensor a, const double s);
#endif

#define NOT_VISITED 0
#define TEMP_VISITED 1
#define PERM_VISITED 2

typedef struct {
  uint32_t type_tag;
  
  Op op; // either NULL or Op. for Tensors op==NULL

  l_list inputs;   // list of Value

  uint8_t shared_data; // if TRUE, not freed...
  t_tensor data;  // lazy eval not supported...

  uint8_t visited;  // for topological sorting

  void *adjoint;    // adjoint node if any: v_bar_i, named out_grad in dlsys world, coerced to Value

  // list of partial adjoints v_bar_k_i, allocated on demand
  // where node k is this node and node i some output node, i.e.
  // a node for which node k is an input...
  l_list node_to_grad;
} Value_struct, *Value;

uint8_t ag_is_value( const void *p);

#define V_SHAPE( v, i) (v)->data->shape[(i)]

void v_dump( const Value v, const uint8_t verbose);

void v_free( void *v);

Value v_negate( const Value v);
Value v_exp( const Value v);
Value v_log( const Value v);
Value v_relu( const Value v);
Value v_relu_deriv( const Value v);

Value v_add_scalar( const Value v1, const double s);
Value v_mul_scalar( const Value v1, const double s);
Value v_div_scalar( const Value v1, const double s);
Value v_power_scalar( const Value v1, const double s);

// element-wise operations
Value v_add( const Value v1, const Value v2);
Value v_sub( const Value v1, const Value v2);
Value v_mul( const Value v1, const Value v2);
Value v_div( const Value v1, const Value v2);
Value v_power( const Value v1, const Value v2);

Value v_sign( const Value u);

Value v_ones( const uint16_t rank, const t_shape shape);
Value v_minus_ones( const uint16_t rank, const t_shape shape);

Value v_matmul( const Value v1, const Value v2);
Value v_transpose( const Value v, const uint16_t axes_len, const t_axes axes);
Value v_reshape( const Value v, const uint16_t rank, const t_shape shape);
Value v_summation( const Value v, const uint16_t axes_len, const t_axes axes,
		   const uint8_t keep_dims);
Value v_broadcast( const Value v, const uint16_t rank, const t_shape shape);

Value v_log_soft_max( const Value Z);

// row-wise summation of exponentiated values and then taking the log.
// input is 2D, output is 1D. rank & axes used in reshaping the gradient to be a 2D
// tensor (1, nbr_rows)...
Value v_log_sum_exp( const Value Z, const uint16_t rank, const uint8_t axes[]);

uint8_t v_is_tensor( const Value v);
Value v_tensor( const t_tensor t);
Value v_tensor_to_value( const t_tensor t, const uint8_t shared_data);

void ag_init();
l_list ag_get_graph( const uint8_t forward);

// see autograd.py:backward()
// out_grad == NULL => allocate a tensor of all ones...: convenience
void ag_gradient( const Value output_tensor, Value out_grad);

void ag_dump( l_list nodes, const uint8_t verbose);

void ag_nn_epoch( const t_tensor X,
		  const t_tensor y,
		  const t_tensor W_1,
		  const t_tensor W_2,
		  const double lr,
		  const unsigned int batch_size,
		  const unsigned int hidden_dim);

Value v_xavier_uniform( const uint32_t fan_in, const uint32_t fan_out, const double gain);
Value v_xavier_normal( const uint32_t fan_in, const uint32_t fan_out, const double gain);
Value v_kaiming_uniform( const uint32_t fan_in, const uint32_t fan_out, const char *non_linearity);
Value v_kaiming_normal( const uint32_t fan_in, const uint32_t fan_out, const char *non_linearity);

// given a 1 D array of y-labels generate a 2 D matrix ( y->shape[0], num_cols)
Value v_one_hot( const uint32_t num_cols, const Value y);
#endif
