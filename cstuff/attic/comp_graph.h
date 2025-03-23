#ifndef _COMP_GRAPH_
#define _COMP_GRAPH_

#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>

#include "d_list.h"

typedef unsigned char boolean;
#define TRUE 1
#define FALSE 0

// reference to a graph node and one of its values: v, v_dot, or v_bar
// { -1, -1} means no variable of any node. used when computing
// adjoint derivatives which just return a result not stored with a graph node
typedef struct {
  int node_idx; // reference to node in cg_set i.e. graph
  char var_idx; // v, v_dot, or v_bar
} cg_f_arg_struct;

// define a function
typedef double (*cg_func) ( const void *cg,
			  const unsigned int nbr_args,
			  const cg_f_arg_struct *args,
			  const cg_f_arg_struct *res);

double cg_identity( const void *cg,
		    const unsigned int nbr_args,
		    const cg_f_arg_struct *args,
		    const cg_f_arg_struct *res);

// to describe an arbitrary function
typedef struct {
  unsigned int nbr_args;  // can be 0
  cg_f_arg_struct *args;  // can be NULL
  cg_f_arg_struct res;    // can be { -1, -1}
  cg_func func;
} cg_func_struct;

// derivative dv_i/dv_k where k is some input node to i
// values are stored with node i for all its parents k
typedef struct {
  unsigned int in_idx; // parent or input node k of node i
  double derivative; // value of adj_derivative
  cg_func_struct adj_derivative; // derivative function dv_i/dv_k
} cg_adjoint_struct, *cg_adjoint;
  
#define V_IDX 0
#define V_DOT_IDX 1
#define V_BAR_IDX 2

#define F_IDX 0
#define F_DOT_IDX 1
#define F_BAR_IDX 2

#define V(n) { n, 0}
#define V_DOT(n) { n, 1}
#define V_BAR(n) { n, 2}
#define NO_RES {-1,-1}  // adjoint derivatives store no result into nodes

typedef struct _cg_node {
  unsigned int idx;  // into cg_set
  void *parents;  // cg_set, input nodes
  void *children; // cg_set, output nodes

  double values[3]; // V, V_DOT, V_BAR

  cg_func_struct funcs[3];  // unused for F_BAR_IDX, empty slot...

  d_list adjoints;  // list of cg_adjoint, cardinality is # of parents
  
  d_list node_to_grad; // list of double, node_to_grad[k]: cardinality is # of children

} cg_node_struct, *cg_node;

// structure to hold a bunch of nodes
typedef struct {
  unsigned int sz;
  unsigned int cnt;
  cg_node *nodes;
} cg_set_struct, *cg_set;

cg_set cg_new( const unsigned int sz, const unsigned int cnt);
void cg_free( cg_set cg);

cg_node cg_get( const cg_set cg, const unsigned int i);

void cg_set_val( const cg_set cg, const int node_idx, const char v_idx, const double v);
double cg_get_val( const cg_set cg, const int node_idx, const char v_idx);

// to set a function of a graph node.
// v_idx: 0..2
void cg_set_func( const cg_set cg,
		  const int node_idx,
		  const char v_idx, 
		  const unsigned int nbr_args,
		  const cg_f_arg_struct *args,
		  const cg_func func);
		  

// to link nodes contained in graph cg
void cg_link( const cg_set cg,
	      const unsigned int out_idx,  // generating output
	      const unsigned int in_idx);  // receiving input

// forward evaluation
// i: starting node from which eval starts
void cg_eval( const cg_set cg, const unsigned int i, const unsigned char f_idx);

// to set for node node_idx (i) the derivative functions dv_i/dv_k where
// k refers to the input nodes i.e. parents...
// we have one function per incoming edge at node_idx
// also creates an empty set of node_to_grad 
void cg_set_adjoint_derivatives( const cg_set cg,
				 const int node_idx,
				 const unsigned int nbr_adjoints, // == nbr of parents
				 cg_adjoint_struct *adjoints);

// reverse automatic derivation
void cg_reverse_AD( const cg_set cg);

#endif
