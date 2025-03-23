#include "comp_graph.h"

#include <assert.h>
#include <math.h>
#include <string.h>

static cg_node new_node( const unsigned int idx) {
  cg_node n = (cg_node) calloc( 1, sizeof( cg_node_struct));
  n->idx = idx;
  // all other values are 0.
  return n;
}

static void copy_func_struct( cg_func_struct *dst, const cg_func_struct *src) {
  dst->nbr_args = src->nbr_args;
  dst->args = (cg_f_arg_struct *) calloc( src->nbr_args, sizeof( cg_f_arg_struct));
  memcpy( dst->args, src->args, src->nbr_args*sizeof( cg_f_arg_struct));
  dst->res = src->res;
  dst->func = src->func;
}

static cg_adjoint cg_adjoint_new( const unsigned int in_idx,
				  const cg_func_struct *derivative) {
  cg_adjoint a = (cg_adjoint) calloc( 1, sizeof( cg_adjoint_struct));
  a->in_idx = in_idx;
  copy_func_struct( &a->adj_derivative, derivative);
  return a;

}

// to check if node idx n_idx is contained in set cg_set
static unsigned char node_set_contains_node( const cg_set ns,
					     const unsigned int n_idx) {
  for ( unsigned int i = 0 ; i < ns->cnt; i++) {
    const cg_node n = ns->nodes[i];
    if ( n->idx == n_idx)
      return TRUE;
  }
  return FALSE;
}

// to set for node node_idx (i) the derivative functions dv_i/dv_k where
// k refers to the input nodes i.e. parents...
// we have one function per incoming edge at node_idx
// also creates an empty set of node_to_grad, i.e. a list of double
void cg_set_adjoint_derivatives( const cg_set cg,
				 const int node_idx,
				 const unsigned int nbr_adjoints, // == nbr of parents
				 cg_adjoint_struct *adjoints) {
  const cg_node n = cg_get( cg, node_idx);

  if ( n->parents != NULL) {
    const unsigned int nbr_in_nodes = ((cg_set) n->parents)->cnt;
    if ( nbr_in_nodes != nbr_adjoints) {
      fprintf( stderr, "cg_set_adjoint_derivatives: node %d: the # of in-nodes mismatches the # of adjoints: %d %d\n",
	       node_idx, nbr_in_nodes, nbr_adjoints);
      exit( -1);
    }
    
    n->adjoints = dl_new( nbr_in_nodes, T_PTR);
    for ( int i = 0; i < nbr_in_nodes; i++) {
      const unsigned int in_idx = adjoints[i].in_idx;
      // check that in_idx is in parents... 
      if ( !node_set_contains_node( (cg_set) n->parents, in_idx)) {
	fprintf( stderr, "cg_set_adjoint_derivatives: node %d: the set of in-nodes (parents) does not contain node %d\n",
		 node_idx, in_idx);
	exit( -1);
      }
      cg_adjoint a = cg_adjoint_new( in_idx, &adjoints[i].adj_derivative);
      dl_append( n->adjoints, (d_list_el) ((void *)a));
    }
  } else {
    if ( nbr_adjoints != 0) {
      fprintf( stderr, "cg_set_adjoint_derivatives: node %d: no in-nodes, # of adjoints != 0: %d\n",
	       node_idx, nbr_adjoints);
      exit( -1);
    }
  }

  if ( n->children != NULL) {
    const unsigned int nbr_out_nodes = ((cg_set) n->children)->cnt;
    n->node_to_grad = dl_new( nbr_out_nodes, T_DOUBLE);
  } else {  // the out (last) node...
    n->node_to_grad = dl_new( 1, T_DOUBLE);
  }
}


#define FREE( p) if ( p != NULL) free( p)

static void free_node( cg_node n) {

  cg_set p = (cg_set) n->parents;
  
  if ( p != NULL) {
    FREE( p->nodes);
  }
  FREE( p);

  cg_set c = (cg_set) n->children;
  if ( c != NULL) {
    FREE( c->nodes);
  }
  FREE( c);

  for ( int i = 0; i <= V_BAR_IDX; i++) {
    cg_func_struct *f = &n->funcs[i];
    FREE( f->args);
  }

  if ( n->adjoints != NULL) {
    dl_free( n->adjoints);
  }
  if ( n->node_to_grad != NULL) {
    dl_free( n->node_to_grad);
  }

  FREE( n);
}

// allocates space fo sz nodes and if cnt > 0 allocates new nodes into
// cg->nodes, otherwise cg->nodes contains NULL ptr
cg_set cg_new( unsigned int sz, unsigned int cnt) {
  assert( sz >= cnt);
  cg_set cg = (cg_set) calloc( 1, sizeof( cg_set_struct));
  cg->sz = sz;
  cg->cnt = cnt;
  cg->nodes = (cg_node *) calloc( sz, sizeof( cg_node));
  for ( unsigned int i = 0; i < cg->cnt; i++) {
    cg->nodes[i] = new_node( i);
  }
  return cg;
}

void cg_free( cg_set cg) {
  for ( int i = 0; i < cg->cnt; i++) {
    free_node( cg->nodes[i]);
  }
  free( cg->nodes);
  free( cg);
}

cg_node cg_get( const cg_set cg, unsigned int i) {
  assert( i < cg->cnt);
  return (cg_node) cg->nodes[i];
}


void cg_set_val( const cg_set cg, const int node_idx,
		 const char v_idx, const double v) {
  assert( node_idx < cg->cnt);
  assert( v_idx <= V_BAR_IDX);
  assert( node_idx >= 0 && v_idx >= 0);

  cg_node n = cg->nodes[node_idx];
  n->values[v_idx] = v;
}

double cg_get_val( const cg_set cg, const int node_idx,
		 const char v_idx) {
  assert( node_idx < cg->cnt);
  assert( v_idx <= V_BAR_IDX);
  assert( node_idx >= 0 && v_idx >= 0);

  cg_node n = cg->nodes[node_idx];
  return n->values[v_idx];
}

static void set_func_struct( cg_func_struct *f,
			     const unsigned int nbr_args, const cg_f_arg_struct *args,
			     cg_f_arg_struct res,
			     const cg_func func) {
  f->nbr_args = nbr_args;
  f->args = (cg_f_arg_struct *) calloc( nbr_args, sizeof( cg_f_arg_struct));
  memcpy( f->args, args, nbr_args * sizeof( cg_f_arg_struct));
  f->res = res;
  f->func = func;
}

void cg_set_func( const cg_set cg,
		  const int node_idx,
		  const char f_idx,
		  const unsigned int nbr_args,
		  const cg_f_arg_struct *args,
		  const cg_func func) {
  assert( node_idx < cg->cnt && node_idx >= 0);
  cg_node n = cg->nodes[node_idx];

  assert( f_idx <= V_BAR_IDX && f_idx >= 0);
  
  cg_func_struct *fp = &(n->funcs[f_idx]);
  // the result is for this node and the given var/func index
  cg_f_arg_struct res = { node_idx, f_idx };

  set_func_struct( fp, nbr_args, args, res, func);
}

typedef double (*nullary_func)();
typedef double (*unary_func) (const double u);
typedef double (*binary_func) (const double u, const double v);

static double n_one() { return 1.0;}
static double n_minus_one() { return -1.0;}

static double u_identity( const double u) {
  return u;
}
static double u_one_over_arg( const double x) {
  return 1.0/x;
}

static double exec_unary_func( const void *cg,
		       const unsigned int nbr_args,
		       const cg_f_arg_struct *args,
		       const cg_f_arg_struct *res,
		       const unary_func func) {
  assert( nbr_args == 1);

  int node_idx = args[0].node_idx;
  char var_idx = args[0].var_idx;
  const double v = cg_get_val( (cg_set) cg, node_idx, var_idx);

  node_idx = res[0].node_idx;
  var_idx = res[0].var_idx;
  double r = (*func)(v);

  if ( node_idx >= 0 && var_idx >= 0)
    cg_set_val( (cg_set) cg, node_idx, var_idx, (*func) (v));

  return r;
}

static double exec_nullary_func( const void *cg,
		       const unsigned int nbr_args,
		       const cg_f_arg_struct *args,
		       const cg_f_arg_struct *res,
		       const nullary_func func) {
  assert( nbr_args == 0);

  const int node_idx = res[0].node_idx;
  const int var_idx = res[0].var_idx;
  double r = (*func)();

  if ( node_idx >= 0 && var_idx >= 0)
    cg_set_val( (cg_set) cg, node_idx, var_idx, (*func) ());

  return r;
}

double cg_one( const void *cg, const unsigned int nbr_args,
	       const cg_f_arg_struct *args,
	       const cg_f_arg_struct *res) {
  return exec_nullary_func( cg, nbr_args, args, res, n_one);
}

double cg_minus_one( const void *cg, const unsigned int nbr_args,
	       const cg_f_arg_struct *args,
	       const cg_f_arg_struct *res) {
  return exec_nullary_func( cg, nbr_args, args, res, n_minus_one);
}

double cg_identity( const void *cg,
		    const unsigned int nbr_args,
		    const cg_f_arg_struct *args,
		    const cg_f_arg_struct *res) {
  return exec_unary_func( cg, nbr_args, args, res, u_identity);
}

double cg_log( const void *cg,
	       const unsigned int nbr_args,
	       const cg_f_arg_struct *args,
	       const cg_f_arg_struct *res) {
  return exec_unary_func( cg, nbr_args, args, res, log);
}

double cg_cos( const void *cg,
	       const unsigned int nbr_args,
	       const cg_f_arg_struct *args,
	       const cg_f_arg_struct *res) {
  return exec_unary_func( cg, nbr_args, args, res, cos);  
}

double cg_sin( const void *cg,
	       const unsigned int nbr_args,
	       const cg_f_arg_struct *args,
	       const cg_f_arg_struct *res) {
  return exec_unary_func( cg, nbr_args, args, res, sin);  
}

double cg_one_over( const void *cg,
	       const unsigned int nbr_args,
	       const cg_f_arg_struct *args,
	       const cg_f_arg_struct *res) {
  return exec_unary_func( cg, nbr_args, args, res, u_one_over_arg);  
}

static double exec_binary_func(
			     const void *cg,
			     const unsigned int nbr_args,
			     const cg_f_arg_struct *args,
			     const cg_f_arg_struct *res,
			     const binary_func func
			     ){
  assert( nbr_args == 2);

  // two args
  int in_node_idx = args[0].node_idx;
  char in_var_idx = args[0].var_idx;
  const double v = cg_get_val( (cg_set) cg, in_node_idx, in_var_idx);
  
  in_node_idx = args[1].node_idx;
  in_var_idx = args[1].var_idx;
  const double w = cg_get_val( (cg_set) cg, in_node_idx, in_var_idx);

  const int out_node_idx = res[0].node_idx;
  const char out_var_idx = res[0].var_idx;

  const double r = (*func) (v, w);
  if ( out_node_idx >= 0 && out_var_idx >= 0)
    cg_set_val( (cg_set) cg, out_node_idx, out_var_idx, r); // hurrah

  return r;
}

static double b_add( const double u, const double v) {
  return u+v;
}
static double b_sub( const double u, const double v) {
  return u-v;
}
static double b_mul( const double u, const double v) {
  return u*v;
}
static double b_div( const double u, const double v) {
  return u/v;
}

static double b_sin_dot( const double u, const double v) {
  return u * cos(v);
}

double cg_add( const void *cg,
	       const unsigned int nbr_args,
	       const cg_f_arg_struct *args,
	       const cg_f_arg_struct *res) {
  return exec_binary_func( cg, nbr_args, args, res, b_add);
}

double cg_sub( const void *cg,
	       const unsigned int nbr_args,
	       const cg_f_arg_struct *args,
	       const cg_f_arg_struct *res) {
  return exec_binary_func( cg, nbr_args, args, res, b_sub);
}

double cg_mul( const void *cg,
	       const unsigned int nbr_args,
	       const cg_f_arg_struct *args,
	       const cg_f_arg_struct *res) {
  return exec_binary_func( cg, nbr_args, args, res, b_mul);
}

double cg_div( const void *cg,
	       const unsigned int nbr_args,
	       const cg_f_arg_struct *args,
	       const cg_f_arg_struct *res) {
  return exec_binary_func( cg, nbr_args, args, res, b_div);
}

double cg_sin_dot( const void *cg,
	       const unsigned int nbr_args,
	       const cg_f_arg_struct *args,
	       const cg_f_arg_struct *res) {
  return exec_binary_func( cg, nbr_args, args, res, b_sin_dot);
}

double cg_add_dot( const void *cg,
	       const unsigned int nbr_args,
	       const cg_f_arg_struct *args,
	       const cg_f_arg_struct *res) {
  return exec_binary_func( cg, nbr_args, args, res, b_add);
}

double cg_sub_dot( const void *cg,
	       const unsigned int nbr_args,
	       const cg_f_arg_struct *args,
	       const cg_f_arg_struct *res) {
  return exec_binary_func( cg, nbr_args, args, res, b_sub);
}

double cg_prod_dot( const void *cg,
	       const unsigned int nbr_args,
	       const cg_f_arg_struct *args,
	       const cg_f_arg_struct *res) {
  assert( nbr_args == 4);

  double v[nbr_args];
  
  for ( int i = 0; i < nbr_args; i++) {
    const int n_idx = args[i].node_idx;
    const char v_idx = args[i].var_idx;
    
    v[i] = cg_get_val( (cg_set) cg, n_idx, v_idx);
  }

  double w = v[0]*v[1] + v[2]*v[3];

  if ( res->node_idx >= 0 && res->var_idx >= 0)
    cg_set_val( (cg_set) cg, res->node_idx, res->var_idx, w);

  return w;
}

static void *check_slots( void *s) {
  cg_set cg = (cg_set) s;
  if ( cg == NULL) {
    return cg_new( 5, 0);  // room for five nodes, but left empty
  }
  if ( cg->cnt == cg->sz) {
    // need to realloc
    const unsigned int new_sz = cg->sz+5;
    cg_node *nn = (cg_node *) calloc( new_sz, sizeof( cg_node));
    memcpy( nn, cg->nodes, sizeof( cg_node) * cg->cnt);
    free( cg->nodes);
    cg->nodes = nn;
    cg->sz = new_sz;
    // count remains unchanged
    return cg;
  }
  return cg;
}

static void link_nodes( const cg_node n_out,  // source: output
			const cg_node n_in) { // dest: input
  

  n_out->children = check_slots( n_out->children);
  n_in->parents = check_slots( n_in->parents);

  cg_set out_children = (cg_set) n_out->children;
  cg_set in_parents = (cg_set) n_in->parents;

  out_children->nodes[out_children->cnt++] = (void *) n_in;
  in_parents->nodes[in_parents->cnt++] = (void *) n_out;
  
}

void cg_link( const cg_set cg,
	      const unsigned int out,  // generating output
	      const unsigned int in) {  // receiving input
  assert( in < cg->cnt);
  assert( out < cg->cnt);
  cg_node n_in = cg->nodes[in];
  cg_node n_out = cg->nodes[out];

  link_nodes( n_out, n_in);
}

void cg_dump( const cg_set cg) {
  for ( int i = 0; i < cg->cnt; i++) {
    const cg_node n = cg->nodes[i];
    fprintf( stderr, "node %d: [%f %f %f], children: [", n->idx, n->values[0], n->values[1], n->values[2]);

    const cg_set children = (cg_set) n->children;
    if ( children != NULL) {
      for ( int j = 0; j < children->cnt; j++) {
	const cg_node m = children->nodes[j];
	fprintf( stderr, "%d,", m->idx);
      }
    }
    fprintf( stderr, "], parents: [");
    const cg_set parents = (cg_set) n->parents;
    if ( parents != NULL) {
      for ( int j = 0; j < parents->cnt; j++) {
	const cg_node m = parents->nodes[j];
	fprintf( stderr, "%d,", m->idx);
      }
    }
    fprintf( stderr, "]\n");    
  }
}


void cg_eval( const cg_set cg, const unsigned int start_node, const unsigned char f_idx) {

  assert( f_idx <= F_BAR_IDX);
  
  for ( unsigned int i = start_node; i < cg->cnt; i++) {
    const cg_node n = cg->nodes[i];
    cg_func_struct *f = &n->funcs[f_idx];

    (f->func) ( cg, f->nbr_args, f->args, &f->res);
  }
}


// for node i search in-node k in adjoints. return NULL if not found
static cg_adjoint get_adjoint( const cg_node node_i, const unsigned int node_k_idx) {

  for ( unsigned int i = 0; i < node_i->adjoints->cnt; i++) {
    const cg_adjoint adj = (cg_adjoint) (dl_get( node_i->adjoints, i)).ptr;
    if ( adj->in_idx == node_k_idx)
      return adj;
  }
  return NULL;
}

static double exec_derivative( const void *cg, const cg_func_struct *f) {
  return (*f->func) (cg, f->nbr_args, f->args, &f->res);
}

void cg_reverse_AD( const cg_set cg) {

  // reset node_to_grad in all nodes
  for ( int i = 0; i < cg->cnt; i++) {
    cg_node node = cg_get( cg, i);
    dl_reset( node->node_to_grad);
  }
  
  cg_node out = cg_get( cg, cg->cnt-1);
  assert( out->children == NULL);
  dl_append( out->node_to_grad, (d_list_el) 1.0);

  for ( int i = cg->cnt-1; i >= 0; i--) {
    const cg_node node = cg_get( cg, i);
    double sum_node_to_grad = dl_sum( node->node_to_grad);
    cg_set_val( cg, i, V_BAR_IDX, sum_node_to_grad);

    const cg_set inputs = node->parents;
    if ( inputs != NULL) {
      for ( int j = 0; j < inputs->cnt; j++) {
	const cg_node node_k = (cg_node) inputs->nodes[j]; // in-node, parent
	double v_k_i_bar = node->values[V_BAR_IDX]; // v_i_bar; cg_get_val( cg, i, V_BAR);

	const cg_adjoint adjoint = get_adjoint( node, node_k->idx);
	if ( adjoint == NULL) {
	  fprintf( stderr, "cg_reverse_AD: can't find adjoint for in-node %d of node %d\n",
		   node_k->idx, node->idx);
	  exit( -1);
	}

	// now exec the derivative func
	// fprintf( stderr, "node i: %d in-node k: %d\n", node->idx, node_k->idx);
	const double dv_i_dv_k = exec_derivative( cg, &(adjoint->adj_derivative));
	v_k_i_bar *= dv_i_dv_k;
	// and append to node_to_grad
	dl_append( node_k->node_to_grad, (d_list_el) v_k_i_bar);
      }
    }
  }
  
}


void main( int argc, char **argv) {

  cg_set cg = cg_new( 7, 7);

  {
    cg_f_arg_struct args[] = { V(0) }; // {{0, V_IDX}};
    cg_set_func( cg, 0, 0, 1, args, cg_identity);
  }

  {
    cg_f_arg_struct args[] = { V(1) };   // {1, V_IDX}};
    cg_set_func( cg, 1, 0, 1, args, cg_identity);
  }

  {
    cg_f_arg_struct args[] = { V(0) }; // , V_IDX}}; // node 0
    cg_set_func( cg, 2, 0, 1, args, cg_log);
  }
 
  {
    cg_f_arg_struct args[] = { V(0), V(1)}; // node 0, 1
    cg_set_func( cg, 3, 0, 2, args, cg_mul);
  }

  {
    cg_f_arg_struct args[] = { V(1) }; //  {1, V_IDX}}; // node 1
    cg_set_func( cg, 4, 0, 1, args, cg_sin);
  }

  {
    cg_f_arg_struct args[] = { V(2), V(3) }; // {2, V_IDX}, {3, V_IDX}}; // node 2, 3
    cg_set_func( cg, 5, 0, 2, args, cg_add);
  }

  {
    cg_f_arg_struct args[] = { V(5), V(4) }; // {5, V_IDX}, {4, V_IDX}}; // node 5, 4
    cg_set_func( cg, 6, 0, 2, args, cg_sub);
  }

  // create graph
  cg_link( cg, 0, 2);
  cg_link( cg, 0, 3);
  cg_link( cg, 1, 3);
  cg_link( cg, 1, 4);
  cg_link( cg, 2, 5);
  cg_link( cg, 3, 5);
  // note: the order here is important for the subtraction
  // v7 = v6 - v5  
  cg_link( cg, 5, 6);
  cg_link( cg, 4, 6);

  cg_dump( cg);

  cg_set_val( cg, 0, V_IDX, 2.0);
  cg_set_val( cg, 1, V_IDX, 5.0);

  cg_eval( cg, 2, F_IDX);

  cg_dump( cg);

  cg_set_val( cg, 0, V_DOT_IDX, 1);
  cg_set_val( cg, 1, V_DOT_IDX, 0);

  {
    cg_f_arg_struct args[] = { V_DOT(0), V(0) };
    cg_set_func( cg, 2, V_DOT_IDX, 2, args, cg_div);
  }

  {
    cg_f_arg_struct args[] = { V_DOT(0), V(1), V_DOT(1), V(0) };
    cg_set_func( cg, 3, V_DOT_IDX, 4, args, cg_prod_dot);
  }

  {
    cg_f_arg_struct args[] = { V_DOT(1), V(1) };
    cg_set_func( cg, 4, V_DOT_IDX, 2, args, cg_sin_dot);
  }
  
  {
    cg_f_arg_struct args[] = { V_DOT(2), V_DOT(3) };
    cg_set_func( cg, 5, V_DOT_IDX, 2, args, cg_add_dot);
  }

  {
    cg_f_arg_struct args[] = { V_DOT(5), V_DOT(4) };
    cg_set_func( cg, 6, V_DOT_IDX, 2, args, cg_sub_dot);
  }

  cg_eval( cg, 2, F_DOT_IDX);
  cg_dump( cg);
  
  cg_set_val( cg, 0, V_DOT_IDX, 0);
  cg_set_val( cg, 1, V_DOT_IDX, 1);
  cg_eval( cg, 2, F_DOT_IDX);
  cg_dump( cg);


  {
    cg_adjoint_struct adjoints[] = {
      { 5, 0.0, {0, NULL, NO_RES, cg_one}},      // dv_7/dv_6
      { 4, 0.0, {0, NULL, NO_RES, cg_minus_one}} // dv_7/dv_5
    };
    cg_set_adjoint_derivatives( cg, 6, 2, adjoints);
  }

  {
    cg_adjoint_struct adjoints[] = {
      { 2, 0.0, {0, NULL, NO_RES, cg_one}},  // dv_6/dv_3
      { 3, 0.0, {0, NULL, NO_RES, cg_one}}   // dv_6/dv_4
    };
    cg_set_adjoint_derivatives( cg, 5, 2, adjoints);
  }

  {
    cg_f_arg_struct args[] = { V(1)};
    cg_adjoint_struct adjoints[] = {
      { 1, 0.0, {1, args, NO_RES, cg_cos}},  // dv_5/dv_2
    };
    cg_set_adjoint_derivatives( cg, 4, 1, adjoints);
  }

  {
    cg_f_arg_struct args_0[] = { V(0)};
    cg_f_arg_struct args_1[] = { V(1)};
    cg_adjoint_struct adjoints[] = {
      { 1, 0.0, {1, args_0, NO_RES, cg_identity}},  // dv_4/dv_2 == v_1
      { 0, 0.0, {1, args_1, NO_RES, cg_identity}},  // dv_4/dv_1 == v_2
    };
    cg_set_adjoint_derivatives( cg, 3, 2, adjoints);
  }

  {
    cg_f_arg_struct args_0[] = { V(0)};
    cg_adjoint_struct adjoints[] = {
      { 0, 0.0, {1, args_0, NO_RES, cg_one_over}},  // dv_3/dv_1 == 1/v_1
    };
    cg_set_adjoint_derivatives( cg, 2, 1, adjoints);
  }

  // the initial nodes have no in-nodes k (parents) and thus no partial adjoints
  // however we call cg_set_adjoint_derivatives to initialize the node_to_grad list
  cg_set_adjoint_derivatives( cg, 1, 0, NULL);
  cg_set_adjoint_derivatives( cg, 0, 0, NULL);  

  cg_reverse_AD( cg);
  cg_dump( cg);
  
  cg_free( cg);
}
