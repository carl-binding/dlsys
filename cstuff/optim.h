#ifndef _OPTIM_H_
#define _OPTIM_H_

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>

#include "list.h"

#define O_TYPE_TAG 0x24242424

#define O_SGD_SUB_TYPE 1
#define O_ADAM_SUB_TYPE 2

typedef struct {
  uint32_t type_tag;
  l_list params;  // list of all parameters of model
  uint32_t sub_type;
} o_optimizer_struct, *o_optimizer;

uint8_t o_is_optimizer( const void *p);

void o_free( o_optimizer o);

void o_step( const o_optimizer o);

void o_reset_grad( const o_optimizer o);

typedef struct {
  o_optimizer_struct super;

  float lr;
  float momentum;
  float weight_decay;

  // list of tensors; one for each model parameter
  // initial value is zeros and then updated...
  l_list u; 
} o_sgd_struct, *o_sgd;

o_sgd o_sgd_new( const l_list parameters, const float lr,
		 const float momentum, const float weight_decay);


typedef struct {
  o_optimizer_struct super;

  float lr;
  float beta1;
  float beta2;
  float eps;
  float weight_decay;
  uint32_t t;

  // list of tensors; one for each model parameter
  // initial value is zeros and then updated...
  l_list m;
  l_list v;
  
} o_adam_struct, *o_adam;

o_adam o_adam_new( const l_list parameters,
		   const float lr,
		   const float beta1,
		   const float beta2,
		   const float eps,
		   const float weight_decay);

#endif
