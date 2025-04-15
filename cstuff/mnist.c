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
#include "list.h"
#include "mnist.h"
#include "logger.h"

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
    log_msg( LOG_ERROR, "failure to read uint32_t\n");
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
    log_msg( LOG_ERROR, "failure to open %s\n", fn_buf);
    return NULL;
  }

  const uint32_t magic_nbr = get_uint32( f);
  if ( magic_nbr != MAGIC_NBR_LABELS) {
    log_msg( LOG_ERROR, "bad magic label nbr: %0x\n", magic_nbr);
    fclose( f);
    return NULL;
  }

  const uint32_t nbr_items = get_uint32( f);
  if ( nbr_items == INT_MAX) {
    log_msg( LOG_ERROR, "nbr of items seems bad... %d\n", nbr_items);
    fclose( f);
    return NULL;
  }

  log_msg( LOG_TRACE, "label data: %d\n", nbr_items);
  
  uint8_t *blob = MEM_CALLOC( nbr_items, sizeof( uint8_t));

  if ( fread( (void *) blob, sizeof( uint8_t), nbr_items, f) != nbr_items) {
    log_msg( LOG_ERROR, "failure to read data\n");
    fclose( f);
    return NULL;
  }

  log_msg( LOG_TRACE, "read %d labels...\n", nbr_items);
  
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
    log_msg( LOG_ERROR, "failure to open %s\n", fn_buf);
    return NULL;
  }

  const uint32_t magic_nbr = get_uint32( f);
  if ( magic_nbr != MAGIC_NBR_IMAGES) {
    log_msg( LOG_ERROR, "bad magic images nbr %0x \n", magic_nbr);
    fclose( f);
    return NULL;
  }

  const uint32_t nbr_items = get_uint32( f);
  if ( nbr_items == INT_MAX) {
    log_msg( LOG_ERROR, "nbr of items seems bad... %d\n", nbr_items);
    fclose( f);
    return NULL;
  }
  
  const uint32_t nbr_rows = get_uint32( f);
  if ( nbr_rows == INT_MAX) {
    log_msg( LOG_ERROR, "nbr of rows seems bad...\n");
    fclose( f);
    return NULL;
  }

  const uint32_t nbr_cols = get_uint32( f);
  if ( nbr_cols == INT_MAX) {
    log_msg( LOG_ERROR, "nbr of cols seems bad...\n");
    fclose( f);
    return NULL;
  }

  log_msg( LOG_TRACE, "image data: (%d, %d, %d)\n", nbr_items, nbr_rows, nbr_cols);

  const uint64_t nbr_bytes = nbr_items * nbr_rows * nbr_cols;
  
  uint8_t *blob = MEM_CALLOC( nbr_bytes, sizeof( uint8_t));

  if ( fread( (void *) blob, sizeof( uint8_t), nbr_bytes, f) != nbr_bytes) {
    log_msg( LOG_ERROR, "failure to read data\n");
    fclose( f);
    return NULL;
  }

  log_msg( LOG_TRACE, "read %ld image bytes...\n", nbr_bytes);
  
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

// from 3D into 2D: (nbr_images, width*height)
t_tensor mnist_flatten_images( t_tensor t) {
  // t_shape_struct t_shape;
  // t_get_shape( t, &t_shape);
  
  assert( t_rank( t) == 3);

  uint32_t shape[2];
  shape[0] = SHAPE_DIM( t, 0);  // nbr of images
  shape[1] = SHAPE_DIM( t, 1) * SHAPE_DIM( t, 2); // rows * cols
  
  // in-situ reshape...
  t_tensor tt = t_reshape( t, 2, shape, t);
  assert( t_rank( tt) == 2);
  return tt;
}

#if 0

// generates a random sequence [0..n-1]
// https://en.wikipedia.org/wiki/Fisher_Yates_shuffle
t_tensor random_permutation( const uint32_t n, const uint8_t dtype) {
  assert( dtype == T_INT64 || dtype == T_INT32);
  // create a 1D array of dtype
  t_tensor t = t_new_vector( n, dtype, NULL);

  int64_t *lp = (int64_t *) t->data;
  int32_t *ip = (int32_t *) t->data;

  // fill with range of 0..n-1
  for ( uint32_t i = 0; i < n; i++) {
    switch( dtype) {
    case T_INT64:
      *lp++ = (int64_t) i;
      break;
    case T_INT32:
      *ip++ = (int32_t) i;
      break;
    default:
      assert( FALSE);
    }
  }

  // run through and permute values randomly
  lp = (int64_t *) t->data;
  ip = (int32_t *) t->data;

  for (uint32_t i = n-1; i >= 0; --i){
    // generate a random number [0, i-1]
    int64_t j = rand() % (i+1);

    //swap the last element with element at random index
    switch ( dtype) {
    case T_INT64:
      {
	int64_t temp = lp[i];
	lp[i] = lp[j];
	lp[j] = temp;
      }
      break;
    case T_INT32:
      {
	int32_t temp = ip[i];
	ip[i] = ip[j];
	ip[j] = temp;
      }
      break;
    }
  }
  
  return t;
}

#endif


data_random_flip_horizontal data_random_flip_horizontal_new( const float p) {
  data_random_flip_horizontal fh = (data_random_flip_horizontal) MEM_CALLOC( 1, sizeof( data_random_flip_horizontal_struct));
  data_transform dt = (data_transform) fh;
  dt->type_tag = DATA_TRANSFORM_TAG;
  dt->sub_class = DATA_RANDOM_FLIP_HORIZONTAL;

  assert( p >= 0 && p <= 1.0);
  
  fh->p = p;

  return fh;
}

data_random_crop data_random_crop_new( const uint32_t padding) {
  data_random_crop rc = (data_random_crop) MEM_CALLOC( 1, sizeof( data_random_crop_struct));
  data_transform dt = (data_transform) rc;
  dt->type_tag = DATA_TRANSFORM_TAG;
  dt->sub_class = DATA_RANDOM_CROP;

  rc->padding = padding;

  return rc;
}

static uint64_t get_img_offset( const t_tensor img,
				const uint32_t i,
				const uint32_t j,
				const uint32_t k) {
  assert( RANK( img) == 3 || RANK( img));
  return t_get_off( img, i, j, k, 0, 0, FALSE);
}

// re-uses given image tensor img, dimensions don't change
static t_tensor random_crop( const data_random_crop rc,
			     t_tensor img) {
  const uint32_t rx = (uint32_t) get_uniform( -rc->padding, rc->padding+1.0);
  const uint32_t ry = (uint32_t) get_uniform( -rc->padding, rc->padding+1.0);

  assert( RANK( img) == 3);

  uint32_t shape[3] = { img->shape[0] + 2*rc->padding,
			img->shape[1] + 2*rc->padding,
			img->shape[2]};

  // allocating an image for padding, with 2*padding margins
  // all zeros
  t_tensor p_img = t_new( RANK( img), shape, DTYPE( img));

  const uint32_t nbr_rows = img->shape[0];
  const uint32_t nbr_cols = img->shape[1];
  const uint32_t depth = img->shape[2];

  // padding the image into p_img by copying
  for ( uint32_t i = 0; i < nbr_rows; i++) {
      for ( uint32_t j = 0; j < nbr_cols; j++) {
	for ( uint32_t k = 0; k < depth; k++) {

	  const uint64_t src_off = get_img_offset( img, i, j, k);
	  const uint64_t dst_off = get_img_offset( img, i+rc->padding, j+rc->padding, k);
	  
	  switch ( DTYPE( img)) {
	  case T_FLOAT:
	    {
	      float *src = (float *) img->data;
	      float *dst = (float *) p_img->data;
	      dst[dst_off] = src[src_off];
	    }
	    break;
	  case T_DOUBLE:
	    {
	      double *src = (double *) img->data;
	      double *dst = (double *) p_img->data;
	      dst[dst_off] = src[src_off];
	    }
	    break;
	  default:
	    assert( FALSE);
	  }
	}
      }
    }
  
  // copying the padded & shifted image with cropping
  // we copy back into original image
  for ( uint32_t i = 0; i < nbr_rows; i++) {
    for ( uint32_t j = 0; j < nbr_cols; j++) {
      for ( uint32_t k = 0; k < depth; k++) {

	const uint64_t dst_off = get_img_offset( img, i, j, k);
	const uint64_t src_off = get_img_offset( img, i+rx, j+ry, k);

	switch ( DTYPE( img)) {
	  case T_FLOAT:
	    {
	      float *dst = (float *) img->data;
	      float *src = (float *) p_img->data;
	      dst[dst_off] = src[src_off];
	    }
	    break;
	  case T_DOUBLE:
	    {
	      double *dst = (double *) img->data;
	      double *src = (double *) p_img->data;
	      dst[dst_off] = src[src_off];
	    }
	    break;
	  default:
	    assert( FALSE);
	} // switch
      } // for k
    } // for j
  } // for i

  t_free( p_img);
  return img;
}

static t_tensor random_flip_horizontal( const data_random_flip_horizontal rf,
					t_tensor img) {
  const double r = get_uniform( 0.0, 1.0);

  if ( r < rf->p) {  // flip image horizontally

    assert( RANK( img) == 2);
    assert( DTYPE( img) == T_FLOAT || DTYPE( img) == T_DOUBLE);


    const uint32_t nbr_cols = T_N_COLS( img);
    const uint32_t nbr_rows = T_N_ROWS( img);
    
    for ( uint32_t i = 0; i < nbr_rows; i++) {
      for ( uint32_t j = 0; j < (uint32_t) nbr_cols/2; j++) {

	uint64_t off_1 = get_img_offset( img, i, j, 0);
	uint64_t off_2 = get_img_offset( img, i, nbr_cols-j-1, 0);
	
	switch ( DTYPE( img)) {
	case T_FLOAT:
	  {
	    float *fp = (float *) img->data;
	    float temp = fp[off_1];
	    fp[off_1] = fp[off_2];
	    fp[off_2] = temp;	    
	  }
	  break;
	case T_DOUBLE:
	  {
	    double *fp = (double *) img->data;
	    double temp = fp[off_1];
	    fp[off_1] = fp[off_2];
	    fp[off_2] = temp;	    
	  }
	  break;
	default:
	  assert( FALSE);
	}
      }
    }
    return img;
  } else {
    return img;
  }
}

			     
t_tensor dt_call( data_transform transform, t_tensor img) {

  data_transform dt = transform;
  
  assert( dt->type_tag == DATA_TRANSFORM_TAG);
  switch ( dt->sub_class) {
  case DATA_RANDOM_CROP:
    return random_crop( (data_random_crop) transform, img);
  case DATA_RANDOM_FLIP_HORIZONTAL:
    return random_flip_horizontal( (data_random_flip_horizontal) transform, img);
  default:
    assert( FALSE);
  }
  
}

mnist_dataset mnist_dataset_new( const char *fn_images,
				   const char *fn_labels,
				   const l_list transforms) {
  mnist_dataset s = (mnist_dataset) MEM_CALLOC( 1, sizeof( mnist_dataset_struct));
  s->type_tag = MNIST_DATASET_TAG;
  s->transforms = transforms;
  s->images = mnist_get_images( fn_images, TRUE); // scaled
  s->images = mnist_flatten_images( s->images); // in-situ
  s->labels = mnist_get_labels( fn_labels);
  assert( t_is1D( s->labels));
  assert( t_is2D( s->images));
  assert( t_1D_len( s->labels) == t_2D_nbr_rows( s->images));
  return s;
}


void mnist_dataset_free( mnist_dataset s) {
  t_free( s->images);
  t_free( s->labels);
  l_free( s->transforms);
  MEM_FREE( s);
}

uint32_t mnist_dataset_length( const mnist_dataset s) {
  return s->labels->shape[0];
}

// to get one image of data set s at given index
// images are 28*28 vectors.
// transforms are applied
// the returned tensor is newly allocated...
t_tensor mnist_dataset_get( const mnist_dataset s,
			     const uint32_t index,
			     uint32_t *label) {

  assert( label != NULL);
  assert( s != NULL);
  assert( s->type_tag == MNIST_DATASET_TAG);
  
  *label = (uint32_t) t_1D_get( s->labels, index);

  assert( RANK(s->images) == 2);
  
  const uint32_t index_arr[1] = {index};
  // allocates a new tensor for each image.
  t_tensor img = t_extract( s->images, index_arr, 1, NULL, DTYPE( s->images));
  assert( RANK( img) == 1);
  assert( t_1D_len( img) == 28*28);
  
  if ( s->transforms != NULL) {
    const uint32_t ll = l_len( s->transforms);
    for ( uint32_t i = 0; i < ll; i++) {
      uint32_t shape[2] = {28, 28};
      // from 1D to 2D...
      img = t_reshape( img, 2, shape, img);
      assert( RANK( img) == 2);
      
      data_transform dt = (data_transform) l_get_p( s->transforms, i);
      img = dt_call( dt, img);
      assert( RANK( img) == 2);
      
      shape[0] = 28*28;
      shape[1] = 0;
      
      img = t_reshape( img, 1, shape, img);
      assert( RANK( img) == 1);
    }
  }

  return img;
  
}


mnist_data_loader mnist_data_loader_new( const mnist_dataset dataset,
					 const uint32_t batch_size,
					 const uint8_t shuffle) {
  mnist_data_loader l = (mnist_data_loader) MEM_CALLOC( 1, sizeof( mnist_data_loader_struct));
  l->type_tag = MNIST_DATA_LOADER_TAG;
  l->dataset = dataset;
  l->batch_size = batch_size;
  l->shuffle = shuffle;

  if ( shuffle) {  // randomnized
    l->order = t_random_permutation( mnist_dataset_length( dataset), T_INT32);
  } else { // in-order
    l->order = t_arange( 0.0, mnist_dataset_length( dataset), 1.0, T_INT32);
  }
  assert( l->order != NULL);

  l->index = 0;
  return l;
}

void mnist_data_loader_free( mnist_data_loader l) {
  assert( l->type_tag == MNIST_DATA_LOADER_TAG);
  mnist_dataset_free( l->dataset);
  t_free( l->order);
  MEM_FREE( l);
}


static void copy_image( const t_tensor dst,
			const uint32_t idx_dst,
			const t_tensor src,
			const uint32_t idx_src,
			const uint32_t row_width) {

  assert( RANK(dst) == 2);
  assert( RANK(src) == 2);

  assert( SHAPE(dst)[1] == SHAPE(src)[1]);
  assert( row_width == SHAPE(dst)[1]);

  assert( idx_dst < SHAPE(dst)[0]);
  assert( idx_src < SHAPE(src)[0]);

  assert( DTYPE(src) == DTYPE( dst));
  assert( DTYPE(src) == T_FLOAT || DTYPE(src) == T_DOUBLE);

  const uint32_t off_dst = idx_dst * row_width;
  const uint32_t off_src = idx_src * row_width;

  if ( DTYPE( src) == T_FLOAT) {
    float *fp_dst = (float *) dst->data;
    fp_dst += off_dst;
    float *fp_src = (float *) src->data;
    fp_src += off_src;

    for ( uint32_t i = 0; i < row_width; i++) {
      *fp_dst++ = *fp_src++;
    }


  } else if ( DTYPE( src) == T_DOUBLE) {
    double *dp_dst = (double *) dst->data;
    dp_dst += off_dst;
    double *dp_src = (double *) src->data;
    dp_src += off_src;

    for ( uint32_t i = 0; i < row_width; i++) {
      *dp_dst++ = *dp_src++;
    }

  } else {
    assert( FALSE);
  }
  
}

void mnist_data_loader_reset( const mnist_data_loader l) {
  assert( l->type_tag == MNIST_DATA_LOADER_TAG);
  l->index = 0;
}

int mnist_data_loader_next( const mnist_data_loader l,
			    t_tensor *labels,
			    t_tensor *images) {

  assert( l->type_tag == MNIST_DATA_LOADER_TAG);
  
  const uint32_t dataset_len = mnist_dataset_length( l->dataset);

  uint32_t from_idx = l->index;
  uint32_t to_idx = from_idx + l->batch_size;
  if ( to_idx >= dataset_len)
    to_idx = dataset_len;

  if ( to_idx <= from_idx) // end of data
    return -1;

  assert( labels != NULL && images != NULL);

  const uint32_t nbr_items = to_idx - from_idx;

  uint32_t shape[2] = {0,0};

  // labels sub-tensor
  shape[0] = nbr_items; shape[1] = 0;
  t_tensor lbl = t_new( 1, shape, DTYPE( l->dataset->labels));

  // images sub-tensor
  const uint32_t img_sz = SHAPE( l->dataset->images)[1]; // nbr of pixels in image: width * height
  shape[0] = nbr_items;
  shape[1] = img_sz;
  t_tensor img = t_new( 2, shape, DTYPE( l->dataset->images));

  assert( RANK( l->order) == 1);
  assert( RANK( l->dataset->labels) == 1);
  
  log_msg( LOG_TRACE, "mnist_data_loader_next: %d .. %d\n", from_idx, to_idx);
  
  for ( uint32_t i = from_idx; i < to_idx; i++) {
    // use l->order to get index into images & labels
    const uint32_t ii = (uint32_t) t_1D_get( l->order, i);

    // use index to get label & image
    const double ll = t_1D_get( l->dataset->labels, ii);
    t_1D_set( lbl, i-from_idx, ll);

    // image is a tensor...
    copy_image( img, i-from_idx, l->dataset->images, ii, img_sz);
    
  }

  l->index = to_idx; // bump up index for next iteration

  *labels = lbl;
  *images = img;

  return 1;
  
}

