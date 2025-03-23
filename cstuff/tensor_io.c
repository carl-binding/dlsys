#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <float.h>
#include <stdarg.h>

#include "mem.h"
#include "tensor.h"
#include "tensor_p.h"

#define FALSE 0
#define TRUE 1

// for 2D tensors only...
#define N_ROWS( t) (t)->shape[0]
#define N_COLS( t) (t)->shape[1]

void assign_to_double( t_value *tv, const t_tensor t, const uint64_t off);
void assign_to_value( t_value *tv, const t_tensor t, const uint64_t off, const int8_t dtype);
void assign_to_tensor( const t_tensor t, const uint64_t off, const t_value vv, const uint8_t coerce);

#define F_WRITE( ptr, sz, cnt, f) { int64_t n = fwrite( ptr, sz, cnt, f); assert( n == cnt); }
#define F_READ( ptr, sz, cnt, f) { int64_t n = fread( ptr, sz, cnt, f); assert( n == cnt); }

void t_write( const t_tensor t, FILE *f) {
  assert( t_assert( t));
  assert( f != NULL);

  // using machine dependent byte order
  // is_sub_tensor not written out...
  assert( !t->is_sub_tensor);
  
  F_WRITE( &(t->dtype), sizeof( uint8_t), 1, f);
  F_WRITE( &(t->rank), sizeof( uint16_t), 1, f);

  if ( t->rank > 0) {
    F_WRITE( t->shape, sizeof( uint32_t), t->rank, f);
    F_WRITE( t->strides, sizeof( uint64_t), t->rank, f);
  } else {
    assert( t->size == 1);
  }

  F_WRITE( &(t->size), sizeof( uint64_t), 1, f);
  F_WRITE( t->data, t_dtype_size( t->dtype), t->size, f);
}

t_tensor t_read( FILE *f) {
  assert( f != NULL);

  uint8_t dtype;
  uint16_t rank;
  
  F_READ( &(dtype), sizeof( uint8_t), 1, f);
  F_READ( &(rank), sizeof( uint16_t), 1, f);

#ifdef INLINED_SHAPE
  t_tensor t = alloc_tensor( rank, dtype);
#else
  t_tensor t = MEM_CALLOC( 1, sizeof( t_tensor_struct));
  t->dtype = dtype;
  t->rank = rank;
#endif
  
  if ( t->rank > 0) {

#ifndef INLINED_SHAPE
    t->shape = MEM_CALLOC( sizeof( uint32_t), t->rank);
    t->strides = MEM_CALLOC( sizeof( uint64_t), t->rank);    
#endif
    
    F_READ( t->shape, sizeof( uint32_t), t->rank, f);
    F_READ( t->strides, sizeof( uint64_t), t->rank, f);
  } // else: rank 0
  
  F_READ( &(t->size), sizeof( uint64_t), 1, f);

  if ( t->rank == 0) {
    assert( t->size == 1);
#ifdef SINGLETON_DATA
    t->data = (void *) &(t->singleton_data);
#else
    t->data = MEM_CALLOC( t_dtype_size( t->dtype), t->size);
#endif
  } else {
    // rank > 0
    t->data = MEM_CALLOC( t_dtype_size( t->dtype), t->size);    
  }
  
  F_READ( t->data, t_dtype_size( t->dtype), t->size, f);

  return t;
}


// returns # of chars printed
static int32_t dump_value( const t_tensor t, const uint64_t off) {
  t_value tv;
  tv.dtype = t->dtype;

  // count of chars printed...
  int32_t cnt = 0;
  
  switch ( tv.dtype) {
  case T_INT8:
    {
      const int8_t *cp = (int8_t *) t->data;
      cnt = fprintf( stderr, "%3d", cp[off]);
    }
    break;
  case T_INT16:
    {
      const int16_t *cp = (int16_t *) t->data;
      cnt = fprintf( stderr, "%5d", cp[off]);
    }
    break;
  case T_INT32:
    {
      const int32_t *cp = (int32_t *) t->data;
      cnt = fprintf( stderr, "%10d", cp[off]);
    }
    break;
  case T_INT64:
    {
      const int64_t *cp = (int64_t *) t->data;
      cnt = fprintf( stderr, "%10ld", cp[off]);
    }
    break;
  case T_FLOAT:
    {
      const float *cp = (float *) t->data;
      cnt = fprintf( stderr, "%10.5f", cp[off]);
    }
    break;
  case T_DOUBLE:
    {
      const double *cp = (double *) t->data;
      cnt = fprintf( stderr, "%12.8g", cp[off]);
    }
    break;
  default:
    assert( FALSE);
  }
  return cnt;
}

// convienience
static char *type2str( int t) {
  switch (t) {
  case T_INT8:
    return "T_INT8";
  case T_INT16:
    return "T_INT16";    
  case T_INT32:
    return "T_INT32";    
  case T_INT64:
    return "T_INT64";    
  case T_FLOAT:
    return "T_FLOAT";    
  case T_DOUBLE:
    return "T_DOUBLE";    
  default:
    return "UNKNOWN";
  }
}

#define PRINT_COMMA( i) { if ( (i) > 0) { fprintf( stderr, ", ");}}
  
void dump_shape( const uint16_t rank, const t_shape shape) {
  fprintf( stderr, " (");
  for ( uint32_t i = 0; i < rank; i++) {
    PRINT_COMMA( i);
    fprintf( stderr, "%d", shape[i]);
  }
  fprintf( stderr, ")");  
}

void dump_limits( const uint16_t rank, const uint32_t *l) {
  fprintf( stderr, " (");
  for ( uint32_t i = 0; i < rank; i++) {
    PRINT_COMMA( i);
    fprintf( stderr, "%d", l[i]);
  }
  fprintf( stderr, ")");  
}

// if output line exceeds length, print new-line and indent.
#define LINE_SIZE 80
static uint32_t new_line( uint32_t count, const uint32_t indent) {
  if ( count >= LINE_SIZE) {
    fprintf( stderr, "\n%*s", indent, " ");
    return indent;
  }
  return count;
}

void t_dump( const t_tensor t, const uint8_t with_header, const uint8_t verbose) {

  if ( t == NULL) {
    fprintf( stderr, "NULL tensor...\n");
    return;
  }

  if ( with_header) {
    // type and rank
    fprintf( stderr, "type = %s, rank = %d\n", type2str( t->dtype), t->rank);
  
    if ( t->rank > 0) {
      fprintf( stderr, "shape = ");
      dump_shape( t->rank, t->shape);

      fprintf( stderr, ", strides = (");    
      for ( uint32_t i = 0; i < t->rank; i++){
	PRINT_COMMA( i);
	fprintf( stderr, "%ld", t->strides[i]);
      }
      
      fprintf( stderr, ")\n");
    }
  }

  if ( !verbose)
    return;
  
  // line length counter
  uint32_t count = 0;

  if ( t->rank == 0) {

    dump_value( t, 0);
    fprintf( stderr, "\n");
    
  } else if ( t->rank == 1) {

    fprintf( stderr, "[");

    for ( uint32_t i = 0; i < t->shape[0]; i++) {
      PRINT_COMMA( i);
      count = new_line( count, 2);
      count += dump_value( t, i);
    }

    fprintf( stderr, "]\n");

  } else if ( t->rank == 2) {
    fprintf( stderr, "[\n");

    for ( uint32_t i = 0; i < t->shape[0]; i++) {
      fprintf( stderr, "%*s[", 2, " ");

      for ( uint32_t j = 0; j < t->shape[1]; j++) {

	const uint64_t off = t_get_off( t, i, j, 0, 0, 0, FALSE);

	PRINT_COMMA( j);
	count = new_line( count, 3);
	count += dump_value( t, off);
	
      }
      fprintf( stderr, "%*s]\n", 2, " ");
      count = 0;
    }
    fprintf( stderr, "]\n");
  } else if ( t->rank == 3) {
    uint32_t limits[ RANK_MAX];
    get_limits( t->rank, t->shape, limits, RANK_MAX);

    fprintf( stderr, "[\n");
    for ( uint32_t i = 0; i < limits[0]; i++) {
      fprintf( stderr, "%*s[\n", 2, " ");

      for ( uint32_t j = 0; j < limits[1]; j++) {
	fprintf( stderr, "%*s[", 4, " ");

	for ( uint32_t k = 0; k < limits[2]; k++) {

	  const uint64_t off = t_get_off( t, i, j, k, 0, 0, FALSE);

	  PRINT_COMMA( k);
	  count = new_line( count, 5);
	  count += dump_value( t, off);

	}
	fprintf( stderr, "%*s]\n", 4, " ");
	count = 0;
      }
      fprintf( stderr, "%*s]\n", 2, " ");
      count = 0;
    }
    fprintf( stderr, "]\n");
  } else if ( t->rank == 4) {

    uint32_t limits[ RANK_MAX];
    get_limits( t->rank, t->shape, limits, RANK_MAX);

    fprintf( stderr, "[\n");
    for ( uint32_t i = 0; i < limits[0]; i++) {
      fprintf( stderr, "%*s[\n", 2, " ");

      for ( uint32_t j = 0; j < limits[1]; j++) {
	fprintf( stderr, "%*s[\n", 4, " ");

	for ( uint32_t k = 0; k < limits[2]; k++) {
	  fprintf( stderr, "%*s[", 6, " ");

	  for ( uint32_t l = 0; l < limits[3]; l++) {

	    const uint64_t off = t_get_off( t, i, j, k, l, 0, FALSE);

	    PRINT_COMMA( l);
	    count = new_line( count, 7);
	    count += dump_value( t, off);

	  }
	  fprintf( stderr, "%*s]\n", 6, " ");
	  count = 0;
	}
	fprintf( stderr, "%*s]\n", 4, " ");
	count = 0;
      }
      fprintf( stderr, "%*s]\n", 2, " ");
      count = 0;
    }
    fprintf( stderr, "]\n");

  } else if ( t->rank == 5) {
    assert( FALSE); // TBD
  } else {
    assert( FALSE);
  }
}


void t_dump_head( const t_tensor t, const int32_t n_rows) {
  if ( n_rows == 0) {
    t_dump( t, FALSE, FALSE);
  } else if ( n_rows > 0) {
    t_tensor u = t_head( t, n_rows);
    t_dump( u, FALSE, FALSE);
    T_FREE( u);
  } else {
    t_tensor u = t_tail( t, n_rows);
    t_dump( u, FALSE, FALSE);
    T_FREE( u);
  }
}

