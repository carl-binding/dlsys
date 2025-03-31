#ifndef __MNIST_H__
#define __MNIST_H__

#include "list.h"
#include "tensor.h"

#define MNIST_DIR "../mnist/"

#define MNIST_TEST_LABELS "t10k-labels-idx1-ubyte"
#define MNIST_TEST_IMAGES "t10k-images-idx3-ubyte"
#define MNIST_TRAIN_LABELS "train-labels-idx1-ubyte"
#define MNIST_TRAIN_IMAGES "train-images-idx3-ubyte"

#define MAGIC_NBR_LABELS 0x00000801
#define MAGIC_NBR_IMAGES 0x00000803

// returns a 1D tensor containing class labels
t_tensor mnist_get_labels( const char *fn);

// if scale, adjust values to 0..1
t_tensor mnist_get_images( const char *fn, const unsigned char scale);

t_tensor mnist_flatten_images( t_tensor t);
unsigned int mnist_get_y_max( const t_tensor y);

#define DATA_TRANSFORM_TAG 0x23232323

#define DATA_RANDOM_CROP 1
#define DATA_RANDOM_FLIP_HORIZONTAL 2

typedef struct {
  uint32_t type_tag;
  uint16_t sub_class;
} data_transform_struct, *data_transform;

typedef struct {
  data_transform_struct dt;
  float p;
} data_random_flip_horizontal_struct, *data_random_flip_horizontal;

typedef struct {
  data_transform_struct dt;
  uint32_t padding;
} data_random_crop_struct, *data_random_crop;

data_random_flip_horizontal data_random_flip_horizontal_new( const float p);

data_random_crop data_random_crop_new( const uint32_t padding);

t_tensor dt_call( data_transform transform, t_tensor img);

#define MNIST_DATASET_TAG 0x24242424

typedef struct {
  uint32_t type_tag;
  t_tensor images;
  t_tensor labels;
  l_list transforms; // NULL or list of data_transform
} mnist_dataset_struct, *mnist_dataset;

mnist_dataset mnist_dataset_new( const char *fn_images,
				   const char *fn_labels,
				   const l_list transforms);

void mnist_dataset_free( mnist_dataset s);

uint32_t mnist_dataset_length( const mnist_dataset s);

t_tensor mnist_dataset_get( const mnist_dataset s,
			     const uint32_t index,
			     uint32_t *label);

#define MNIST_DATA_LOADER_TAG 0x25252525

typedef struct {
  uint32_t type_tag;
  mnist_dataset dataset;
  uint32_t batch_size;
  uint8_t shuffle;
  t_tensor order; 
  uint32_t index;
} mnist_data_loader_struct, *mnist_data_loader;

mnist_data_loader mnist_data_loader_new( const mnist_dataset dataset,
					 const uint32_t batch_size,
					 const uint8_t shuffle);

void mnist_data_loader_free( mnist_data_loader l);

int mnist_data_loader_next( const mnist_data_loader l,
			    t_tensor *labels,
			    t_tensor *images);
  
#endif
