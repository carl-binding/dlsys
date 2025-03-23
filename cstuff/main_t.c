#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "tensor.h"
#include "mnist.h"

#define TRUE 1
#define FALSE 0

static double square_func( double a) {
  return a * a;
}

void main( int argc, char *argv) {

  t_tensor sss = t_new_scalar( 232.3, T_FLOAT);
  t_free( sss);
  sss = t_new( 0, NULL, T_FLOAT);
  t_free( sss);
  double ddd[1] = {323.454};
  sss = t_new_tensor( 0, NULL, T_FLOAT, ddd);
  t_free( sss);


#ifdef INC_IDX
  static uint32_t lll[] = { 3, 4, 3, 2};
  static uint32_t ccc[] = { 0, 0, 0, 0};

  while ( inc_idx( 4, ccc, lll)) {
    fprintf( stderr, "%ld %ld %ld %ld\n", ccc[0], ccc[1], ccc[2], ccc[3]);
  }
#endif
  
  const double v[] = { 1.1, 2.2, 3.3};
  const double m[] = { 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9};
  const double mm[] = { 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8,
			9.9, 11.11, 12.12, 13.13, 14.14, 15.15, 16.16, 17.17, 18.18, 19.19};

  const double rr[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23};

  t_tensor t = t_new_matrix( 3, 3, T_FLOAT, rr);
  t_tensor lsm = t_log_softmax( t, NULL);
  t_dump( lsm, TRUE, TRUE);
  t_free( t);
  t_free( lsm);

  t = t_new_matrix( 3, 3, T_FLOAT, rr);
  t_tensor lse = t_log_sumexp( t, NULL);
  t_dump( lse, TRUE, TRUE);
  t_free( t);
  t_free( lse);

  exit( 0);

  t = t_new_matrix( 3, 3, T_FLOAT, rr);
  uint8_t axis[2] = {0,1};
  t_tensor s = t_sum( t, axis, FALSE);
  t_dump( s, TRUE, TRUE);
  t_free( s);
  axis[0] = 1; axis[1] = 0;
  s = t_sum( t, axis, FALSE);
  t_dump( s, TRUE, TRUE);
  t_free( s);
  t_free( t);

  t = t_new_matrix( 3, 3, T_FLOAT, rr);
  t_tensor tt = t_matmul( t, t, NULL);

  t_dump( t, TRUE, TRUE);
  t_dump( tt, TRUE, TRUE);

  T_FREE( tt);
  T_FREE( t);

  // exit( 0);
  
  tt = t_new_vector( 24, T_FLOAT, rr);

  fprintf( stderr, "%d\n", __LINE__);
  const uint32_t shape0[3] = {2, 3, 4};
  t_tensor ttt = t_reshape( tt, 3, shape0, NULL);
  t_dump( ttt, TRUE, TRUE);
  t_free( ttt);

  fprintf( stderr, "%d\n", __LINE__);
  tt = t_reshape( tt, 3, shape0, tt);
  t_dump( tt, TRUE, TRUE);

  // exit( 0);
  
  fprintf( stderr, "%d\n", __LINE__);
  t = t_new_vector( 4, T_FLOAT, rr);
  tt = t_new_vector( 5, T_FLOAT, mm);
  t_dump( t, TRUE, TRUE);
  t_dump( tt, TRUE, TRUE);
  t_tensor ww = t_outer( t, tt, NULL);
  t_dump( ww, TRUE, TRUE);
  T_FREE( tt); T_FREE( t); T_FREE( ww);

  fprintf( stderr, "%d\n", __LINE__);
  t = t_new_vector( 4, T_DOUBLE, rr);
  tt = t_new_vector( 5, T_DOUBLE, mm);
  t_dump( t, TRUE, TRUE);
  t_dump( tt, TRUE, TRUE);
  ww = t_outer( t, tt, NULL);
  t_dump( ww, TRUE, TRUE);
  T_FREE( tt); T_FREE( t); T_FREE( ww);

  fprintf( stderr, "%d\n", __LINE__);
  t = t_new_vector( 4, T_INT64, rr);
  tt = t_new_vector( 5, T_INT64, mm);
  t_dump( t, TRUE, TRUE);
  t_dump( tt, TRUE, TRUE);
  ww = t_outer( t, tt, NULL);
  t_dump( ww, TRUE, TRUE);
  T_FREE( tt); T_FREE( t); T_FREE( ww);

  // exit( 0);

  // inner product (5 x 4 x 3) * (3 x 2 x 3)
  
  fprintf( stderr, "%d\n", __LINE__);  
  tt = t_arange( 0, 60, 1, T_FLOAT);
  uint32_t shape2[3] = {5, 4, 3};
  tt = t_reshape( tt, 3, shape2, NULL);
  t_dump( tt, TRUE, TRUE);

  ww = t_arange( 0, 18, 1, T_FLOAT);
  shape2[0] = 3;
  shape2[1] = 2;
  shape2[2] = 3;
  t_tensor www = t_reshape( ww, 3, shape2, NULL);
  t_dump( www, TRUE, TRUE);
  t_free( www);

  www = t_reshape( ww, 3, shape2, ww);
  t_dump( www, TRUE, TRUE);

  // exit( 0);

  t_tensor vv = t_inner( tt, ww, NULL);
  assert( vv != NULL);
  // out.shape = a.shape[:-1] + b.shape[:-1]
  assert( vv->shape[0] == tt->shape[0]);
  assert( vv->shape[1] == tt->shape[1]);
  assert( vv->shape[2] == ww->shape[0]);
  assert( vv->shape[3] == ww->shape[1]);
  t_dump( vv, TRUE, TRUE);
  T_FREE( ww); T_FREE( tt); T_FREE( vv);
				   
  fprintf( stderr, "%d\n", __LINE__);  
  t = t_new_scalar( 3.09, T_FLOAT);
  t_dump( t, TRUE, TRUE);
  t_tensor u = t_as_type( t, T_INT16);
  t_dump( u, TRUE, TRUE);
  t_free( t); t_free( u);

  fprintf( stderr, "%d\n", __LINE__);  
  t = t_new_vector( 3, T_FLOAT, v);
  t_dump( t, TRUE, TRUE);
  u = t_as_type( t, T_INT16);
  t_dump( u, TRUE, TRUE);
  t_free( t); t_free( u);

  fprintf( stderr, "%d\n", __LINE__);  
  t = t_new_matrix( 3, 3, T_FLOAT, m);
  t_dump( t, TRUE, TRUE);
  u = t_as_type( t, T_INT16);
  t_dump( u, TRUE, TRUE);
  t_free( t); t_free( u);

  fprintf( stderr, "%d\n", __LINE__);  
  uint32_t shape[3] = {3, 3, 2};
  t = t_new_tensor(3, shape, T_FLOAT, mm);
  t_dump( t, TRUE, TRUE);
  u = t_as_type( t, T_INT16);
  t_dump( u, TRUE, TRUE);
  t_free( t); t_free( u);

  uint32_t shape_4[4] = {3, 3, 2, 2};
  const double mmm[] = { 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 11.11, 12.12,
			 13.13, 14.14, 15.15, 16.16, 17.17, 18.18, 19.19,
			 21.21, 22.22, 23.23, 24.24, 25.25, 26.26,
			 27.27, 28.28, 29.29, 111.11, 112.12, 113.13, 114.14, 115.15,
			 116.16, 117.17, 118.18, 119.19,
			 220, 221, 223, 224, 225, 226, 227, 228, 229, 230 };

  fprintf( stderr, "%d\n", __LINE__);  
  t = t_new_tensor( 4, shape_4, T_FLOAT, mmm);
  t_dump( t, TRUE, TRUE);
  u = t_as_type( t, T_INT16);
  t_dump( u, TRUE, TRUE);
  t_free( t); t_free( u);

  const double m12[] = { 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10, 11.11, 12.12};
  t = t_new_matrix( 3, 4, T_FLOAT, m12);
  t_dump( t, TRUE, TRUE);

  fprintf( stderr, "%d\n", __LINE__);  
  uint32_t shape12[2] = {2, 6};
  u = t_reshape( t, 2, shape12, NULL);
  t_dump( u, TRUE, TRUE);

  t_free( t); t_free( u);

  t = t_new_tensor(3, shape, T_FLOAT, mm);
  t_dump( t, TRUE, TRUE);

  // exit( 0);
  
  fprintf( stderr, "%d\n", __LINE__);  
  uint32_t shape13[2] = {2, 9};
  u = t_reshape( t, 2, shape13, NULL);
  t_dump( u, TRUE, TRUE);
  t_free( u);

  fprintf( stderr, "%d\n", __LINE__);    
  uint32_t shape14[2] = {9, 2};
  u = t_reshape( t, 2, shape14, NULL);
  t_dump( u, TRUE, TRUE);
  t_free( u);

  fprintf( stderr, "%d\n", __LINE__);    
  t = t_new_matrix( 3, 3, T_FLOAT, m);
  t_dump( t, TRUE, TRUE);
  u = t_apply( t, square_func, NULL);
  t_dump( u, TRUE, TRUE);
  t_free( u);

  fprintf( stderr, "%d\n", __LINE__);    
  u = t_as_type( t, T_INT16);
  vv = t_apply( u, square_func, NULL);
  t_dump( vv, TRUE, TRUE);
  t_free( u); t_free( vv);

  // exit( 0);
  
  // broadcasting
  t = t_new_vector( 10, T_FLOAT, m12);
  u = t_new_matrix( 32, 10, T_FLOAT, NULL);

  fprintf( stderr, "%d\n", __LINE__);    
  t_dump( t, TRUE, TRUE); t_dump( u, TRUE, TRUE);
  
  if ( t_broadcastable( t, u)) {
    fprintf( stderr, "broadcastable ok\n");
  } else {
    fprintf( stderr, "broadcastable NOT ok\n");
    assert( FALSE);
  }

  fprintf( stderr, "%d\n", __LINE__);    
  u = t_new_matrix( 17, 13, T_FLOAT, NULL);
  if ( !t_broadcastable( t, u)) {
    fprintf( stderr, "broadcastable ok\n");    
  } else {
    fprintf( stderr, "broadcastable NOT ok\n");
    assert( FALSE);
  }

  // vector op scalar
  t = t_new_vector( 10, T_FLOAT, m12);
  u = t_new_scalar( 4.0, T_FLOAT);
  
  fprintf( stderr, "%d\n", __LINE__);
  t_dump( t, TRUE, TRUE);
  t_dump( u, TRUE, TRUE);
  t_tensor w = t_add( t, u, NULL);
  t_dump( w, TRUE, TRUE);
  T_FREE( w);

  fprintf( stderr, "%d\n", __LINE__);
  t_dump( t, TRUE, TRUE);
  t_dump( u, TRUE, TRUE);
  w = t_subtract( t, u, NULL);
  t_dump( w, TRUE, TRUE);
  T_FREE( w);

  fprintf( stderr, "%d\n", __LINE__);
  t_dump( t, TRUE, TRUE);
  t_dump( u, TRUE, TRUE);
  w = t_multiply( t, u, NULL);
  t_dump( w, TRUE, TRUE);
  T_FREE( w);
  
  fprintf( stderr, "%d\n", __LINE__);
  t_dump( t, TRUE, TRUE);
  t_dump( u, TRUE, TRUE);
  w = t_divide( t, u, NULL);
  t_dump( w, TRUE, TRUE);
  T_FREE( w);

  T_FREE( t); T_FREE( u);

  exit( 0);
    
  t = t_new_vector( 10, T_FLOAT, m12);
  u = t_new_matrix( 32, 10, T_FLOAT, NULL);
  t_dump( t, TRUE, TRUE); t_dump( u, TRUE, TRUE);

  fprintf( stderr, "%d\n", __LINE__);    
  // element wise add
  w = t_add( u, t, NULL);
  t_dump( w, TRUE, TRUE);

  t = t_new_scalar( 3.09, T_FLOAT);
  t_dump( t, TRUE, TRUE);
  u = t_new_scalar( 4.2, T_FLOAT);
  t_dump( u, TRUE, TRUE);

  fprintf( stderr, "%d\n", __LINE__);    
  // dot scalar, scalar
  w = t_dot( t, u, NULL);
  t_dump( w, TRUE, TRUE);
  t_free( w); t_free( u);

  // dot scalar, vector
  u = t_new_vector( 10, T_FLOAT, m12);

  fprintf( stderr, "%d\n", __LINE__);    
  t_dump( t, TRUE, TRUE);
  t_dump( u, TRUE, TRUE);
  w = t_dot( t, u, NULL);
  fprintf( stderr, "t * u = \n");
  t_dump( w, TRUE, TRUE);
  t_free( w);

  fprintf( stderr, "%d\n", __LINE__);    
  // dot vector, scalar
  t_dump( u, TRUE, TRUE);
  t_dump( t, TRUE, TRUE);
  w = t_dot( u, t, NULL);
  fprintf( stderr, "u * t = \n");
  t_dump( w, TRUE, TRUE);
  t_free( w);

  fprintf( stderr, "%d\n", __LINE__);    
  // dot vector, vector of same length
  t_dump( u, TRUE, TRUE);
  w = t_dot( u, u, NULL);
  fprintf( stderr, "u * u = \n");
  if ( w == NULL) {
    fprintf( stderr, "t_dot: FAIL\n");
    assert( FALSE);
  } else {
    fprintf( stderr, "t_dot: OK\n");
    t_dump( w, TRUE, TRUE);
    t_free( w);
  }

  fprintf( stderr, "%d\n", __LINE__);    
  // dot vector, vector different lengths
  t = t_new_vector( 9, T_FLOAT, m12);
  t_dump( u, TRUE, TRUE);
  t_dump( t, TRUE, TRUE);
  w = t_dot( u, t, NULL);
  fprintf( stderr, "u * t = \n");
  if ( w != NULL) {
    fprintf( stderr, "t_dot: FAIL\n");
    assert( FALSE);    
  } else {
    fprintf( stderr, "t_dot: OK\n");
  }

  fprintf( stderr, "%d\n", __LINE__);    
  t_dump( u, TRUE, TRUE);
  t_dump( t, TRUE, TRUE);
  w = t_dot( t, u, NULL);
  fprintf( stderr, "t * u = \n");
  if ( w != NULL) {
    fprintf( stderr, "t_dot: FAIL\n");
    assert( FALSE);    
  } else {
    fprintf( stderr, "t_dot: OK\n");
  }

  // dot matrix
  t = t_new_matrix( 3, 4, T_FLOAT, mmm);
  u = t_new_matrix( 3, 4, T_FLOAT, mmm);

  fprintf( stderr, "%d\n", __LINE__);    
  // dot matrix, (3 x 4) x (3 x 4): mismatching shapes
  t_dump( u, TRUE, TRUE);
  t_dump( t, TRUE, TRUE);
  w = t_dot( t, u, NULL);

  fprintf( stderr, "t * u = \n");
  if ( w != NULL) {
    fprintf( stderr, "t_dot: FAIL\n");
    assert( FALSE);    
  } else {
    fprintf( stderr, "t_dot: OK\n");
  }

  fprintf( stderr, "%d\n", __LINE__);    
  u = t_new_matrix( 4, 3, T_FLOAT, mmm);
  // dot matrix, (3 x 4) x (4 x 3): matching shapes
  t_dump( t, TRUE, TRUE);
  t_dump( u, TRUE, TRUE);
  w = t_dot( t, u, NULL);
  fprintf( stderr, "t * u = \n");
  if ( w == NULL) {
    fprintf( stderr, "t_dot: FAIL\n");
    assert( FALSE);    
  } else {
    fprintf( stderr, "t_dot: OK\n");
    t_dump( w, TRUE, TRUE);
    t_free( w);
  }

  fprintf( stderr, "%d\n", __LINE__);    
  // tensor x vector, length along last axis differ...
  // dot( a, b), b 1D, a 2D, last axis differ
  t = t_new_matrix( 3, 4, T_FLOAT, mmm);
  u = t_new_vector( 5, T_FLOAT, mmm);

  t_dump( t, TRUE, TRUE);
  t_dump( u, TRUE, TRUE);
  
  w = t_dot( t, u, NULL);
  fprintf( stderr, "t * u = \n");
  if ( w != NULL) {
    fprintf( stderr, "t_dot: FAIL\n");
    assert( FALSE);    
  } else {
    fprintf( stderr, "t_dot: OK\n");
  }
  T_FREE( u);

  fprintf( stderr, "%d\n", __LINE__);    
  // dot( a, b), b 1D, a 2D, last axis match
  // this is the frequent case for NN...
  u = t_new_vector( 4, T_FLOAT, mmm);
  t = t_new_matrix( 3, 4, T_FLOAT, mmm);
  
  t_dump( t, TRUE, TRUE);
  t_dump( u, TRUE, TRUE);

  w = t_dot( t, u, NULL);
  fprintf( stderr, "t * u = \n");
  if ( w == NULL) {
    fprintf( stderr, "t_dot: FAIL\n");
    assert( FALSE);
  } else {
    fprintf( stderr, "t_dot: OK\n");
    t_dump( w, TRUE, TRUE);
    t_free( w);
  }
  T_FREE( t);
  T_FREE( u);
  

  fprintf( stderr, "%d\n", __LINE__);    
  uint32_t shape10[1] = { 784 };
  u = t_rand( 1, shape10, T_FLOAT);
  uint32_t shape11[2] = { 30, 784 };
  t = t_rand( 2, shape11, T_FLOAT);

  fprintf( stderr, "%d\n", __LINE__);      
  w = NULL;
  for ( int i = 0; i < 5000; i++) {
    w = t_dot( t, u, w);
    if ( w == NULL) {
      fprintf( stderr, "t_dot: FAIL\n");
      assert( FALSE);
    } else {
      // fprintf( stderr, "t_dot: OK\n");
    }
  }
  T_FREE( w);
  
  fprintf( stderr, "done with T_FLOAT dot\n");

  fprintf( stderr, "%d\n", __LINE__);    
  u = t_rand( 1, shape10, T_FLOAT);
  t = t_rand( 2, shape11, T_FLOAT);
  for ( int i = 0; i < 5000; i++) {
    w = t_dot( t, u, w);
    if ( w == NULL) {
      fprintf( stderr, "t_dot: FAIL\n");
      assert( FALSE);
    } else {
      // fprintf( stderr, "t_dot: OK\n");
    }
  }

  fprintf( stderr, "done with T_FLOAT dot\n");

  fprintf( stderr, "%d\n", __LINE__);    
  // dot( a, b), b 1D, a 3D, last axis match in length
  uint32_t shape15[3] = { 3, 3, 4};  // u is 1D len == 4
  t = t_new_tensor( 3, shape15, T_FLOAT, mmm);
  u = t_new_vector( 4, T_FLOAT, mmm);
  t_dump( t, TRUE, TRUE);
  t_dump( u, TRUE, TRUE);
  
  w = t_dot( t, u, NULL);
  fprintf( stderr, "t * u = \n");
  if ( w == NULL) {
    fprintf( stderr, "t_dot: FAIL\n");
    assert( FALSE);
  } else {
    fprintf( stderr, "t_dot: OK\n");
    t_dump( w, TRUE, TRUE);
    t_free( w);
  }
  T_FREE( t);
  
  fprintf( stderr, "%d\n", __LINE__);    
  uint32_t ss[] = { 4, 1, 5};
  t = t_new_tensor( 3, ss, T_FLOAT, NULL);
  w = t_new_scalar( 3, T_FLOAT);
  t_dump( t, TRUE, TRUE);
  t = t_add( t, w, NULL);
  t_dump( t, TRUE, TRUE);
  t_free( t);

  
  fprintf( stderr, "%d\n", __LINE__);    
  t = t_new_tensor( 3, ss, T_FLOAT, NULL);  
  t = t_subtract( t, w, NULL);
  t_dump( t, TRUE, TRUE);
  t_free( t);
  
  fprintf( stderr, "%d\n", __LINE__);    
  t = t_new_tensor( 3, ss, T_FLOAT, NULL);  
  t = t_multiply( t, w, NULL);
  t_dump( t, TRUE, TRUE);
  t_free( t);
  
  fprintf( stderr, "%d\n", __LINE__);    
  t = t_new_tensor( 3, ss, T_FLOAT, NULL);  
  t = t_multiply( t, w, NULL);
  t_dump( t, TRUE, TRUE);
  t_free( t);
  
  ss[0] = 4; ss[1] = 1; ss[2] = 0;

  fprintf( stderr, "%d\n", __LINE__);
  u = t_new_tensor( 2, ss, T_FLOAT, NULL);
  w = t_new_scalar( 4, T_FLOAT);
  u = t_add( u, w, NULL);
  t_free( w);

  t_dump( u, TRUE, TRUE);
  t_dump( t, TRUE, TRUE);

  fprintf( stderr, "%d\n", __LINE__);    
  // that should be broadcastable. rank( t) > rank( u)
  assert( t_broadcastable( u, t));
  // t_dump( u, TRUE, TRUE);
  // t_dump( t, TRUE, TRUE);
  w = t_add( u, t, NULL);
  t_dump( w, TRUE, TRUE);
  t_free( w);

  t_free( u);
  t_free( t);

  fprintf( stderr, "%d\n", __LINE__);      
  t = t_new_matrix( 4, 4, T_FLOAT, mmm);
  uint8_t axes[2] = { 1, 0};
  t_dump( t, TRUE, TRUE);
  u = t_max( t, axes, FALSE);
  t_dump( u, TRUE, TRUE);
  T_FREE( u);

  fprintf( stderr, "%d\n", __LINE__);      
  u = t_abs( t, NULL);
  t_dump( u, TRUE, TRUE);
  T_FREE( u);

  fprintf( stderr, "%d\n", __LINE__);      
  u = t_log( t, NULL);
  t_dump( u, TRUE, TRUE);
  T_FREE( u);

  fprintf( stderr, "%d\n", __LINE__);      
  u = t_sign( t, NULL);
  t_dump( u, TRUE, TRUE);
  T_FREE( u);

  fprintf( stderr, "%d\n", __LINE__);      
  t_dump( t, TRUE, TRUE);
  u = t_transpose( t, NULL);
  t_dump( u, TRUE, TRUE);
  T_FREE( u);

  T_FREE( t);

  fprintf( stderr, "%d\n", __LINE__);      
  ss[0] = ss[1] = ss[2] = 1;
  t = t_new_tensor( 3, ss, T_FLOAT, mmm);
  t_dump( t, TRUE, TRUE);
  t = t_squeeze( t, NULL);
  t_dump( t, TRUE, TRUE);
  T_FREE( t);

  fprintf( stderr, "%d\n", __LINE__);      
  t = t_new_tensor( 1, ss, T_FLOAT, mmm);
  t_dump( t, TRUE, TRUE);
  t = t_squeeze( t, NULL);
  t_dump( t, TRUE, TRUE);
  T_FREE( t);
  
  fprintf( stderr, "%d\n", __LINE__);
  ss[0] = ss[1] = 1; ss[2] = 3;
  t = t_new_tensor( 3, ss, T_FLOAT, mmm);
  t_dump( t, TRUE, TRUE);
  t = t_squeeze( t, NULL);
  t_dump( t, TRUE, TRUE);
  T_FREE( t);

  fprintf( stderr, "%d\n", __LINE__);
  ss[0] = 1; ss[1] = 4; ss[2] = 3;
  t = t_new_tensor( 3, ss, T_FLOAT, mmm);
  t_dump( t, TRUE, TRUE);
  t = t_squeeze( t, NULL);
  t_dump( t, TRUE, TRUE);
  T_FREE( t);
  
  fprintf( stderr, "%d\n", __LINE__);
  t = t_new_vector( 3, T_FLOAT, mmm);
  t_dump( t, TRUE, TRUE);
  u = t_diag( t);
  t_dump( u, TRUE, TRUE);
  T_FREE( u);

  fprintf( stderr, "%d\n", __LINE__);
  t_dump( t, TRUE, TRUE);
  u = t_transpose( t, NULL);
  t_dump( u, TRUE, TRUE);

  fprintf( stderr, "%d\n", __LINE__);
  w = t_transpose( u, NULL);
  t_dump( w, TRUE, TRUE);
  T_FREE( u); T_FREE( w);

  t_dump( t, TRUE, TRUE);
  uint16_t reps[2] = {2, 1};
  u = t_tile( t, 2, reps);
  t_dump( u, TRUE, TRUE);
  T_FREE( u);

  fprintf( stderr, "%d\n", __LINE__);
  t_dump( t, TRUE, TRUE);
  reps[0] = 2; reps[1] = 3;
  u = t_tile( t, 2, reps);
  t_dump( u, TRUE, TRUE);
  T_FREE( u);

  fprintf( stderr, "%d\n", __LINE__);
  t = t_new_matrix( 4, 3, T_FLOAT, m12);
  u = t_new_matrix( 4, 3, T_FLOAT, m12);
  t_dump( t, TRUE, TRUE); t_dump( u, TRUE, TRUE);
  w = t_inner( t, u, NULL);
  t_dump( w, TRUE, TRUE);
  T_FREE( w);


  fprintf( stderr, "%d\n", __LINE__);    
  uint32_t argmax_index[1][t->rank];
  t_dump( t, TRUE, TRUE);
  t_argmax( t, -1, (uint32_t *) argmax_index, 1);
  fprintf( stderr, "argmax =[ %d, %d]\n", argmax_index[0][0], argmax_index[0][1]);

  fprintf( stderr, "%d\n", __LINE__);    
  uint32_t argmax_index_2[4][t->rank]; // 4 rows -> 4 indices into 4 x 3 matrix
  t_dump( t, TRUE, TRUE);
  // along row-axis which for 2D tensors is at shape index 0 vertically
  t_argmax( t, 0, (uint32_t *) argmax_index_2, t_2D_nbr_rows( t));
  for ( int i = 0; i < t_2D_nbr_rows( t); i++) {
    fprintf( stderr, "argmax[%d] = [%d, %d]\n", i, argmax_index_2[i][0], argmax_index_2[i][1]);
  }
  t_free( t);

  fprintf( stderr, "%d\n", __LINE__);      
  const double ffff[] = {
    0.13084372,   0.10088878,   0.12315741,  0.094358284,  0.083103412,   0.10580701,   0.11003486, 
    0.1109837,   0.05678081,   0.08404201};
  t = t_new_vector( 10, T_FLOAT, ffff);
  uint32_t argmax_index_3[t->rank];
  t_dump( t, TRUE, TRUE);
  t_argmax( t, -1, (uint32_t *) argmax_index_3, 1);
  fprintf( stderr, "argmax = [ %d]\n", argmax_index_3[0]);
  
// #define RUN_NN

#ifdef RUN_NN

  srand( 1313); // random seed

  const nn_network nn = nn_build_3_4_1( 1.0, 0.0, 0.01);
  nn_dump( nn, TRUE, TRUE);

#define NBR_SAMPLES 4  // size of data & truths
#define BATCH_SIZE  4  // size of batches
  
#define NBR_FEATURES 3 

#define NBR_ITER 1000
  
  const double samples[] = {
     0, 0, 1,
     1, 1, 1,
     1, 0, 1,
     0, 1, 1
  };

  const double truths[] = { 0, 1, 1, 0 };

  t_tensor x_data = t_new_matrix( NBR_SAMPLES, NBR_FEATURES, T_FLOAT, samples);
  t_tensor t_data = t_new_vector( NBR_SAMPLES, T_FLOAT, truths);

  nn_train( nn, x_data, t_data, BATCH_SIZE, NBR_ITER, TRUE);

  exit( 1);

#endif // RUN_NN

  
}
