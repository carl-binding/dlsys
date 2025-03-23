#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <float.h>
#include <stdarg.h>

#include <netinet/in.h>

#include "mem.h"
#include "tensor.h"
#include "mnist.h"

#define FALSE 0
#define TRUE 1

static int get_fn( const char *_fn, char *fn_buf, int fn_buf_sz) {
  memset( fn_buf, 0, fn_buf_sz);
  strncpy( fn_buf, MNIST_DIR, strlen( MNIST_DIR));
  strncpy( fn_buf + strlen( MNIST_DIR), _fn, strlen( _fn));
  return 1;
}

static uint32_t get_uint32( FILE *f) {
  uint32_t i = 0;
  if ( fread( (void *) &i, sizeof( uint32_t), 1, f) != 1) {
    fprintf( stderr, "failure to read uint32_t\n");
    return INT_MAX;
  }
  // data is in MSB, network-byte order....
  i = ntohl( i);
  return i;
}
  
t_tensor mnist_get_labels( const char *fn) {
  char fn_buf[512];
  get_fn( fn, fn_buf, sizeof( fn_buf));
  
  FILE *f = fopen( fn_buf, "r");

  if ( f == NULL) {
    fprintf( stderr, "failure to open %s\n", fn_buf);
    return NULL;
  }

  const uint32_t magic_nbr = get_uint32( f);
  if ( magic_nbr != MAGIC_NBR_LABELS) {
    fprintf( stderr, "bad magic label nbr: %0x\n", magic_nbr);
    fclose( f);
    return NULL;
  }

  const uint32_t nbr_items = get_uint32( f);
  if ( nbr_items == INT_MAX) {
    fprintf( stderr, "nbr of items seems bad... %d\n", nbr_items);
    fclose( f);
    return NULL;
  }

  fprintf( stderr, "label data: %d\n", nbr_items);
  
  uint8_t *blob = MEM_CALLOC( nbr_items, sizeof( uint8_t));

  if ( fread( (void *) blob, sizeof( uint8_t), nbr_items, f) != nbr_items) {
    fprintf( stderr, "failure to read data\n");
    fclose( f);
    return NULL;
  }

  fprintf( stderr, "read %d labels...\n", nbr_items);
  
  t_tensor t = t_new_vector( nbr_items, T_INT8, NULL);

  int8_t *dptr = (int8_t *) t->data;
  for ( int i = 0; i < nbr_items; i++) {

    *dptr++ = blob[i];
    
    // assign_to_tensor( t, i, tv, FALSE);
  }
  
  MEM_FREE( blob);
  fclose( f);
  
  return t;
}

// returns a 3D tensor of nbr_images * nbr_rows * nbr_cols of FLOAT
t_tensor mnist_get_images( const char *fn, const unsigned char scale) {

  char fn_buf[512];
  get_fn( fn, fn_buf, sizeof( fn_buf));
  
  FILE *f = fopen( fn_buf, "r");

  if ( f == NULL) {
    fprintf( stderr, "failure to open %s\n", fn_buf);
    return NULL;
  }

  const uint32_t magic_nbr = get_uint32( f);
  if ( magic_nbr != MAGIC_NBR_IMAGES) {
    fprintf( stderr, "bad magic images nbr %0x \n", magic_nbr);
    fclose( f);
    return NULL;
  }

  const uint32_t nbr_items = get_uint32( f);
  if ( nbr_items == INT_MAX) {
    fprintf( stderr, "nbr of items seems bad... %d\n", nbr_items);
    fclose( f);
    return NULL;
  }
  
  const uint32_t nbr_rows = get_uint32( f);
  if ( nbr_rows == INT_MAX) {
    fprintf( stderr, "nbr of rows seems bad...\n");
    fclose( f);
    return NULL;
  }

  const uint32_t nbr_cols = get_uint32( f);
  if ( nbr_cols == INT_MAX) {
    fprintf( stderr, "nbr of cols seems bad...\n");
    fclose( f);
    return NULL;
  }

  fprintf( stderr, "image data: (%d, %d, %d)\n", nbr_items, nbr_rows, nbr_cols);

  const uint64_t nbr_bytes = nbr_items * nbr_rows * nbr_cols;
  
  uint8_t *blob = MEM_CALLOC( nbr_bytes, sizeof( uint8_t));

  if ( fread( (void *) blob, sizeof( uint8_t), nbr_bytes, f) != nbr_bytes) {
    fprintf( stderr, "failure to read data\n");
    fclose( f);
    return NULL;
  }

  fprintf( stderr, "read %ld image bytes...\n", nbr_bytes);
  
  uint32_t shape[3] = { nbr_items, nbr_rows, nbr_cols };

  const uint8_t dtype = T_FLOAT;
  
  t_tensor t = t_new_tensor( 3, shape, dtype, NULL);

  float *dptr = (float *) t->data;

  t_value tv;
  tv.dtype = dtype;
  
  for ( uint64_t i = 0; i < nbr_bytes; i++) {
    // coerce
    tv.u.f = (float) blob[i];
    // scale to 0..1
    if ( scale)
      tv.u.f /= 255.0;
    // and assign ...
    *dptr++ = tv.u.f;
  }

  MEM_FREE( blob);
  fclose( f);
  
  return t;

}

unsigned int mnist_get_y_max( const t_tensor y) {
  t_tensor y_max = t_max( y, NULL, FALSE);
  const unsigned int m = (unsigned int) t_scalar( y_max);
  t_free( y_max);
  return m;
}

// from 2D into 1D
t_tensor mnist_flatten_images( t_tensor t) {
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
