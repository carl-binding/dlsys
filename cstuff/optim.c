#include <assert.h>
#include <math.h>

#include "optim.h"
#include "mem.h"
#include "tensor.h"
#include "autograd.h"

#define FALSE 0
#define TRUE 1

uint8_t o_is_optimizer( const void *p) {
  const o_optimizer o = (o_optimizer) p;
  return o->type_tag == O_TYPE_TAG;
}

void o_free( o_optimizer o) {

  assert( o_is_optimizer( o));

  l_free( o->params);

  switch( o->sub_type) {
  case O_SGD_SUB_TYPE:
    {
      o_sgd sgd = (o_sgd) o;
      l_free( sgd->u);
    }
    break;
  case O_ADAM_SUB_TYPE:
    {
      o_adam a = (o_adam) o;
      l_free( a->m);
      l_free( a->v);
    }
    break;
  default:
    assert( FALSE);
  }
  MEM_FREE( o);
}

// we check the parameters are a non empty list of values
static uint8_t check_parameters( const l_list parameters) {
  if ( parameters->cnt == 0) {
    fprintf( stderr, "check_parameters: empty list...\n");
    return FALSE;
  }
  if ( parameters->type != T_PTR) {
    fprintf( stderr, "check_parameters: list elements are not T_PTR\n");
    return FALSE;
  }
  for ( uint32_t i = 0; i < parameters->cnt; i++) {
    const Value v = (Value) l_get_p( parameters, i);
    if ( !v_is_value( v)) {
      fprintf( stderr, "check_parameters: list element is not a Value (%d)\n", i);
      return FALSE;
    }
  }
  return TRUE;
}

o_sgd o_sgd_new( const l_list parameters, const float lr,
		 const float momentum, const float weight_decay) {
  o_sgd sgd = MEM_CALLOC( 1, sizeof( o_sgd_struct));
  o_optimizer o = (o_optimizer) sgd;

  assert( parameters != NULL);
  assert( check_parameters( parameters));
  o->type_tag = O_TYPE_TAG;
  o->sub_type = O_SGD_SUB_TYPE;
  o->params = parameters;

  sgd->lr = lr;
  sgd->momentum = momentum;
  sgd->weight_decay = weight_decay;

  // the elements of the u list are tensors, initialized to zeros and updated during
  // stepping
  sgd->u = l_new( parameters->cnt, T_PTR, (void *) (void *) t_free);
  for ( uint32_t i = 0; i < parameters->cnt; i++) {
    const Value v = (Value) l_get_p( parameters, i);
    assert( v_is_value( v));
    t_tensor t = t_zeros( v_rank( v), v_shape( v), v_dtype( v));
    // append at tail, i.e. at index i
    l_append_ptr( sgd->u, t);
  }
  
  return sgd;
}

o_adam o_adam_new( const l_list parameters,
		   const float lr,
		   const float beta1,
		   const float beta2,
		   const float eps,
		   const float weight_decay) {
  return NULL;
}

// sanity check to ensure that we have the adjoint in our global list
// of values...
static uint8_t adjoint_in_bwd_value_list( const Value v) {
  const l_list l = ag_get_val_list( AG_BWD_MODE);
  l_el el;
  el.ptr = (void *) v;
  return l_contains( l, el);
}

void o_step( const o_optimizer o) {

  /*
    for i, param in enumerate(self.params):
            grad = ndl.Tensor(param.grad, dtype='float32').data       
            grad = grad + self.weight_decay * param.data     
            ## grad = clip_grad( grad)
            self.u[i] = self.momentum * self.u[i] + (1 - self.momentum) * grad
            param.data = param.data - self.u[i] * self.lr
   */
  for ( uint32_t i = 0; i < l_len( o->params); i++) {
    if ( o->sub_type == O_SGD_SUB_TYPE) {

      const Value param = (Value) l_get_p( o->params, i);
      assert( v_is_value( param));

      const Value v_adjoint = (Value) param->adjoint;
      assert( v_adjoint != NULL);
      assert( v_is_value( v_adjoint));
      assert( adjoint_in_bwd_value_list( v_adjoint));
      
      t_tensor grad = (t_tensor) v_adjoint->data;
      assert( RANK( grad) == RANK( param->data));

      const o_sgd sgd = (o_sgd) o;

      // newly allocated
      t_tensor param_data_times_weight_decay = t_mul_scalar( param->data, sgd->weight_decay, /*in-situ*/ FALSE);
      // in-situ add
      grad = t_add( grad, param_data_times_weight_decay, grad);
      t_free( param_data_times_weight_decay);

      t_tensor u_i = (t_tensor) l_get_p( sgd->u, i);
      // in-situ
      u_i = t_mul_scalar( u_i, sgd->momentum, TRUE);
      // newly allocated...
      t_tensor tt = t_mul_scalar( grad, (1-sgd->momentum), /*in-situ*/ FALSE);
      // in-situ
      u_i = t_add( u_i, tt, u_i);
      t_free( tt);

      // newly allocated...
      t_tensor ui_times_lr = t_mul_scalar( u_i, sgd->lr, /*in-situ*/ FALSE);
      param->data = t_subtract( param->data, ui_times_lr, param->data);
      t_free( ui_times_lr);
      
    } else if ( o->sub_type == O_ADAM_SUB_TYPE) {
      /*
	self.t += 1
        for i, param in enumerate(self.params):
            grad = ndl.Tensor(param.grad, dtype='float32').data + param.data * self.weight_decay
            # m_{t+1}, v{t+1}
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad**2
            # bias correction
            m_hat = (self.m[i]) / (1 - self.beta1 ** self.t)
            v_hat = (self.v[i]) / (1 - self.beta2 ** self.t)
            param.data = param.data - self.lr * m_hat / (v_hat ** 0.5 + self.eps)
      */
      const Value param = (Value) l_get_p( o->params, i);
      assert( v_is_value( param));

      const Value v_adjoint = (Value) param->adjoint;
      assert( v_adjoint != NULL);
      assert( v_is_value( v_adjoint));
      assert( adjoint_in_bwd_value_list( v_adjoint));
      
      t_tensor grad = (t_tensor) v_adjoint->data;
      assert( RANK( grad) == RANK( param->data));

      const o_adam adam = (o_adam) o;

      adam->t += 1;

      t_tensor m_i = (t_tensor) l_get_p( adam->m, i);
      t_tensor v_i = (t_tensor) l_get_p( adam->v, i);

      double t = 1.0 - adam->beta1;
      t_tensor grad_times_t = t_mul_scalar( grad, t, /* in-situ */ FALSE);
      m_i = t_mul_scalar( m_i, adam->beta1, /* in-situ */ TRUE);
      m_i = t_add( m_i, grad_times_t, m_i);

      t_free( grad_times_t);

      t = 1.0 - adam->beta2;
      t_tensor grad_squared = t_pow_scalar( grad, 2.0, /* in-situ */ FALSE);
      grad_squared = t_mul_scalar( grad_squared, t, /* in-situ */ TRUE);
      v_i = t_mul_scalar( v_i, adam->beta2, /* in-situ */ TRUE);
      v_i = t_add( v_i, grad_squared, v_i);

      t_free( grad_squared);

      t = 1.0 - pow( adam->beta1, adam->t);
      t_tensor m_hat = t_div_scalar( m_i, t, /* in-situ */ FALSE);
      m_hat = t_mul_scalar( m_hat, adam->lr, /* in-situ */ TRUE);

      t = 1.0 - pow( adam->beta2, adam->t);
      t_tensor v_hat = t_div_scalar( v_i, t, /* in-situ */ FALSE);
      v_hat = t_pow_scalar( v_hat, 0.5, /* in-situ */ TRUE);
      v_hat = t_add_scalar( v_hat, adam->eps, /* in-situ */ TRUE);

      t_tensor m_hat_div_v_hat = t_divide( m_hat, v_hat, NULL);

      param->data = t_subtract( param->data, m_hat_div_v_hat, param->data);

      t_free( m_hat);
      t_free( v_hat);
      t_free( m_hat_div_v_hat);
      
    } else {
      assert( FALSE);
    }
			   
  }
}

void o_reset_grad( const o_optimizer o) {

  if ( o == NULL)
    return;

  assert( o_is_optimizer( o));
  for ( uint32_t i = 0; i < l_len( o->params); i++) {
    const Value v = (Value) l_get_p( o->params, i);
    assert( v_is_value( v));
    // the parameter values are *not* part of the global list of values created
    // during backward() computation of gradients.
    // parameters of modules are owned by said modules. However their adjoints are
    // computed during backward computation since parameters are inputs to the module computation...
    // and are thus freed upon resetting the global BWD value list
    
    assert( v->adjoint != NULL);
    assert( adjoint_in_bwd_value_list( v->adjoint));
    v->adjoint = NULL;
    l_reset( v->node_to_grad);
  }
}




