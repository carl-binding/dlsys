#ifndef _COMP_GRAPH_
#define _COMP_GRAPH_

#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>

#include "list.h"

typedef unsigned char boolean;
#define TRUE 1
#define FALSE 0

// define a function
// args: pointers to cg_node from which we retrieve the values as arguments
// returns a double value to be stored within the calling node...
typedef double (*cg_func) ( const unsigned int nbr_args, const void **args);

// to describe an arbitrary function
typedef struct {
  unsigned int nbr_args;  // can be 0
  void **args;  // can be NULL, pointing to cg_node
  cg_func func;
} cg_func_struct;


#define MAX_NAME_LEN 64
typedef struct _cg_node {
  char name[MAX_NAME_LEN];
  l_list inputs;  // input nodes, list of cg_node
  l_list outputs; // output nodes, list of cg_node

  unsigned char evaluated; // to ensure proper traversal...
  
  double value;

  cg_func_struct func;
  
} cg_node_struct, *cg_node;

// we simply use an array of nodes. not efficient to lookup node by names...
l_list cg_new( const unsigned int cnt, const char **node_names);
void cg_free( l_list cg);

cg_node cg_get( const l_list cg, const char *n);

// creates a link from named out-node to named in-node
void cg_link( const l_list cg,
	      const char *from,
	      const char *to);

void cg_set_func( const l_list cg,
		  const char *node_nm,
		  const cg_func func,
		  const unsigned int nbr_args,
		  ...);

// to set the node's value & evaluated flag to TRUE. used for constant & initial values
void cg_set_val( const l_list cg,
		 const char *node_nm,
		 const double v);

double cg_get_val( const l_list cg, const char *node_nm);

// evaluates sub-graph of cg. note that the topological ordering must be
// established at graph construction time...
void cg_eval( const l_list cg,
	      const char *from_node,
	      const char *to_node);
	      


#endif
