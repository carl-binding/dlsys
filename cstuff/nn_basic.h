#ifndef _NN_BASIC_H_
#define _NN_BASIC_H_

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>

#include "list.h"
#include "tensor.h"
#include "autograd.h"

#define MDL_IDENTITY 1
#define MDL_LINEAR 2
#define MDL_FLATTEN 3
#define MDL_RELU 4
#define MDL_SEQUENTIAL 5
#define MDL_SOFTMAX_LOSS 6
#define MDL_BATCH_NORM_1D 7
#define MDL_LAYER_NORM_1D 8
#define MDL_DROPOUT 9
#define MDL_RESIDUAL 10

#define MDL_TYPE_TAG 0x21212121

typedef struct {
  uint32_t type_tag;
  
  uint16_t sub_class;
  
  l_list parameters;  // list of parameters, coerced to t_tensor
  l_list modules;  // children modules, i.e. entries coerce to mdl_module

  uint8_t training; // TRUE -> training, FALSE -> evaluating

  Value (*forward) (const void *module, const uint32_t n_args, ...);
  
} mdl_module_struct, *mdl_module;

// recursively traverses modules to get all parameters
l_list mdl_parameters( const mdl_module m, l_list parameters);

// recursively traverses modules to get all children modules
l_list mdl_children( const mdl_module m, l_list children);

// returns a Tensor-Value. assumes t is a tensor-value
// invokes the module->forward() function
Value mdl_call( const mdl_module m, uint32_t n_args, ...);

void mdl_eval( const mdl_module m);
void mdl_train( const mdl_module m);

void mdl_free( const mdl_module m);

typedef struct {
  mdl_module_struct mdl; // super-class
} mdl_identity_struct, *mdl_identity;

mdl_identity mdl_identity_new();

typedef struct {
  mdl_module_struct mdl; // super-class

  uint32_t in_features;
  uint32_t out_features;

  // parameters
  Value weight;
  Value bias;
  
} mdl_linear_struct, *mdl_linear;

mdl_linear mdl_linear_new( const uint32_t in_features, const uint32_t out_features);

typedef struct {
  mdl_module_struct mdl; // super-class
} mdl_flatten_struct, *mdl_flatten;

typedef struct {
  mdl_module_struct mdl; // super-class
} mdl_relu_struct, *mdl_relu;

mdl_flatten mdl_flatten_new();
mdl_relu mdl_relu_new();

typedef struct {
  mdl_module_struct mdl; // super-class
} mdl_sequential_struct, *mdl_sequential;

mdl_sequential mdl_sequential_new( const uint32_t n_args, ...);

typedef struct {
  mdl_module_struct mdl; // super-class
} mdl_softmax_loss_struct, *mdl_softmax_loss;

mdl_softmax_loss mdl_softmax_loss_new( const uint32_t n_args, ...);

typedef struct {
  mdl_module_struct mdl; // super-class
} mdl_batch_norm1D_struct, *mdl_batch_norm1D;

typedef struct {
  mdl_module_struct mdl; // super-class
} mdl_layer_norm1D_struct, *mdl_layer_norm1D;

typedef struct {
  mdl_module_struct mdl; // super-class
} mdl_dropout_struct, *mdl_dropout;

typedef struct {
  mdl_module_struct mdl; // super-class
} mdl_residual_struct, *mdl_residual;







#endif // _NN_BASIC_H_
