#ifndef __TENSOR_H__
#define __TENSOR_H__

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "mem.h"

// unary ops
#define MAX_OP 1
#define MIN_OP 2
#define SUM_OP 3
#define ABS_OP 4
#define LOG_OP 5
#define EXP_OP 6
#define SIGN_OP 7
#define ASSIGN_OP 8
#define SQUARE_OP 9
#define RELU_OP 10
#define RELU_DERIV_OP 11
#define SIGMOID_OP 12
#define SIGMOID_DERIV_OP 13
#define TANH_OP 14
#define TANH_DERIV_OP 15
#define NEGATE_OP 16

// for range checks...
#define MAX_UNARY_OP NEGATE_OP

// binary ops
#define PLUS_OP 1
#define MINUS_OP 2
#define TIMES_OP 3
#define DIVIDE_OP 4
#define DOT_OP 5

#define EQ_OP 6
#define LT_OP 7
#define GT_OP 8
#define LE_OP 9
#define GE_OP 10

#define POWER_OP 11

// for range checks...
#define MAX_BINARY_OP POWER_OP

// supported data types
#define T_INT8  1
#define T_INT16 3
#define T_INT32 5
#define T_INT64 7
#define T_FLOAT 9
#define T_DOUBLE 11

#define RANK_MAX 5

// length is given by rank...
typedef uint32_t t_shape[];
typedef uint64_t t_strides[];
// we use axes as an array of boolean. index 0 is row-axes, index 1 is column-axes, index 3 is depth-axes, etc.
// this differs from the numpy usage of axes which uses tuples of integers....
typedef uint8_t t_axes[];

// we align data in such a way that we can unroll the dot-product loop in some cases
// and use streaming SIMD extension (SSE2) to our advantage.
// see t_dot_loop_double() in tensor_p.c
#define DOT_LOOP_UNROLL_DOUBLE 2   // 2 doubles of 64 bits == 128 bit
#define DOT_LOOP_UNROLL_FLOAT  4   // 4 floats of 32 bits == 128 bit
#define SSE2_REGISTER_SIZE     16  // 128 bits in bytes...

#define SINGLETON_DATA

// if defined, we inline shape and strides *after* the tensor struct...
// saves us a alloc() and free(), but is a bit of hairy memory layout
// in particular, when the rank of a tensor GROWS, a new t_tensor_struct
// needs to be allocated to have room for larger shape & stride data.
typedef struct {
  uint8_t dtype;      // indicates data type of tensor elements
  uint8_t is_sub_tensor;  // TRUE if it's a subtensor sharing data...
  uint16_t rank;
  uint32_t *shape;    // [rank], # of elements per dimension

  uint64_t *strides;  // [rank], indicates # of elements across higher dimensions
  uint64_t size;      // number of elements in tensor

#ifdef SINGLETON_DATA
  // in case of rank == 0, i.e. a scalar wrapped into a tensor we don't alloc/free the
  // data slot. which is large enough to hold up to 8*8 == 64 bits of double or int64
  uint8_t singleton_data[8];
#endif

  // ensure that data is aligned on an appropriate boundary
  // must be last field in struct.
  uint8_t *data __attribute__ ((aligned (SSE2_REGISTER_SIZE)));
} t_tensor_struct, *t_tensor;

#define DTYPE( t) (t)->dtype
#define SHAPE( t) (t)->shape
#define RANK( t) (t)->rank

// for data-type coercion 
typedef struct {
  uint8_t dtype;
  union {
    int8_t c;
    int16_t s;
    int32_t l;
    int64_t ll;
    float f;
    double d;
  } u;
} t_value;

typedef struct {
  uint16_t rank;
  uint32_t shape[RANK_MAX];
} t_shape_struct;
  
double sigmoid( const double x);
double sigmoid_derivate( const double x);
double tanh_derivate( const double x);
double relu( const double x);
double relu_derivate( const double x);

void assign_to_double( t_value *tv, const t_tensor t, const uint64_t off);
void assign_to_long( t_value *tv, const t_tensor t, const uint64_t off);
void assign_to_value( t_value *tv, const t_tensor t, const uint64_t off, const int8_t dtype);
void assign_to_tensor( const t_tensor t, const uint64_t off, const t_value vv, const uint8_t coerce);

double get_uniform( const double lower, const double upper);

uint64_t t_get_off( const t_tensor t,
		    const uint32_t _i,
		    const uint32_t _j,
		    const uint32_t _k,
		    const uint32_t _l,
		    const uint32_t _m,
		    const uint8_t broadcast);

// nbr of bytes used for various data-types
uint8_t t_dtype_size( const uint8_t t);

uint8_t t_isInt( const t_tensor t);
uint8_t t_isReal( const t_tensor t);

// macro. nbr of elements in tensor
#define t_size(t) ((t)->size)

#ifdef INLINED_SHAPE
t_tensor alloc_tensor(  const uint8_t rank, const uint8_t dtype);
#endif

// returns a 0D tensor of given data-type.
t_tensor t_new_scalar( const double v, const uint8_t dtype);

// zero filled new tensor
t_tensor t_new( const uint16_t rank, const t_shape shape, const uint8_t dtype);

// returns a 1D tensor. v == NULL or len(v) >= len
t_tensor t_new_vector( const uint32_t len, const uint8_t dtype, const double *v);

// returns a 2D tensor, v == NULL or len(v) >= n_rows * n_cols. row-order data layout
t_tensor t_new_matrix( const uint32_t n_rows, const uint32_t n_cols, const uint8_t dtype, const double *data);

// returns an N-D tensor. v == NULL or len( v) >= nbr_elems(t)
t_tensor t_new_tensor( const uint16_t rank, const t_shape shape, const uint8_t dtype, const double *data);

t_tensor t_sub( t_tensor t, const t_shape index, const uint32_t len_index, const uint8_t copy,
		const uint8_t squeeze);

// allocate a new sub-tensor..
t_tensor t_slice( const t_tensor t, uint32_t index[][2], const uint32_t len_index);

// uniform [0..1]
t_tensor t_rand( const uint16_t rank, const t_shape shape, const uint8_t dtype);

t_tensor t_normal( const double mean, const double std_dev,
		   const uint16_t rank, const t_shape shape, const uint8_t dtype);
// convenience
#define t_randn( rank, shape, dtype) t_normal( 0.0, 1.0, rank, shape, dtype)

// binary: probability of success i.e. value == 1
t_tensor t_randb( const double prob, const uint16_t rank, const t_shape shape, const uint8_t dtype);

t_tensor t_ones( const uint16_t rank, const t_shape shape, const uint8_t dtype);
#define t_ones_like( t) t_ones( (t)->rank, (t)->shape, (t)->dtype)

#define t_zeros( rank, shape, dtype) t_new( rank, shape, dtype)
#define t_zeros_like( t) t_new( (t)->rank, (t)->shape, (t)->dtype)

t_tensor t_constant( const double c, const uint16_t rank, const t_shape shape, const uint8_t dtype);

// take a 1 D tensor of labels and turn it into a 2D one-hot tensor
// nbr_cols >= max_value( t) or -1 in which case nbr_cols == max_value(t)
t_tensor t_one_hot( const t_tensor t, const uint8_t dtype, const int32_t nbr_cols);

// more convenience.... generates 2D tensors. dtype == FLOAT or DOUBLE.
#define XAVIER_UNIFORM 1
#define XAVIER_NORMAL 2
t_tensor t_xavier( const uint8_t distribution,
		   const uint32_t fan_in,
		   const uint32_t fan_out,
		   const double gain,
		   const uint8_t dtype);

#define KAIMING_UNIFORM 3
#define KAIMING_NORMAL 4

// non_linearity == "RELU"
t_tensor t_kaiming( const uint8_t distribution,
		    const uint32_t fan_in,
		    const uint32_t fan_out,
		    const char *non_linearity,
		    const uint8_t dtype);


t_tensor t_arange( const double lo, const double hi, const double step, const uint8_t dtype);

// create a one-hot 1D tensor (vector)  given a value and the length of the vector == nbr of labels
t_tensor t_to_one_hot_vector( const uint32_t value, const uint32_t len_vector,
			      const t_tensor out, const uint8_t dtype);

t_tensor t_nan_to_num( const t_tensor t, const uint8_t copy, const double nan,
		       const double pos_inf, const double neg_inf);

void t_clear( const t_tensor t);

// fills the given tensor with value v, coercing if needed.
void t_fill( t_tensor t, const double v);

// tests if two tensors have same rank and shape.
uint8_t t_same_shape( const t_tensor t, const t_tensor u);

// tests if tensor t's shape match the given args, terminated with -1
// value == 0 causes shape dimension to be ignored
uint8_t t_check_shape( const t_tensor t, ...);

#define SHAPE_DIM(t, i) (t)->shape[i]

// declares a shape struct named t_shape and initializes it.
#define GET_SHAPE( t) t_shape_struct t##_shape; t_get_shape(t, &(t##_shape));

uint8_t t_equal( const t_tensor t, const t_tensor u);

// test if all values are zero
uint8_t t_all_zero( const t_tensor t);

uint8_t t_has_nan( const t_tensor t);
uint8_t t_has_inf( const t_tensor t);

// to verify that values in tensor t lie in range [val_min..val_max]
uint8_t t_check_values( const t_tensor t, const double val_min, const double val_max);

// copies data of tensor src into tensor dst, assuming similar rank and shape
// returns destination tensor. no new data structures are allocated.
// returns NULL in case of non-matching shape or rank
// if necessary, data-type coercion is performed
t_tensor t_copy( t_tensor dst, const t_tensor src);

// returns a newly allocated copy of t
t_tensor t_clone( t_tensor t);

// type coercion. returns a newly allocated data-structure.
t_tensor t_as_type( const t_tensor s, const uint8_t dtype);

// some sanity check on tensor struct
uint8_t t_assert( const t_tensor t);

// get tensor's rank
uint16_t t_rank( const t_tensor t);

uint8_t t_is0D( const t_tensor t);
// get 0D tensor value as double
double t_scalar( const t_tensor t);

uint8_t t_is1D( const t_tensor t);
// get 1D tensor value at given position as double
double t_1D_get( const t_tensor t, const uint32_t off);

// sets 1D tensor value at given position, handling coercion.
void t_1D_set( const t_tensor t, const uint32_t off, const double v);

// returns length of 1D tensor
uint32_t t_1D_len( const t_tensor t);

uint8_t t_is2D( const t_tensor t);
// get 2D tensor value at given position as double
double t_2D_get( const t_tensor t, const uint32_t row, const uint32_t col);

// sets 2D tensor value at given position, handling coercion.
void t_2D_set( const t_tensor t, const uint32_t row, const uint32_t col, const double v);
void t_2D_set_tv( const t_tensor t, const uint32_t i, const uint32_t j, const t_value tv);

t_tensor t_as_matrix( const t_tensor t, const uint8_t tranpose);

uint32_t t_2D_nbr_rows( const t_tensor t);
uint32_t t_2D_nbr_cols( const t_tensor t);

#define T_N_ROWS( t) t_2D_nbr_rows( t)
#define T_N_COLS( t) t_2D_nbr_cols( t)

// to get the value at given offsets. assumes that len( offsets) >= rank( t)
// returns value coerced to double. offsets into tensor dimensions
t_value t_get_tv( const t_tensor t, const uint32_t *offsets);
double t_get( const t_tensor t, const uint32_t *offsets);

// to set the value at given offsets. assumes that len( offsets) >= rank( t)
// coerces value from double to data-type of tensor t.
void t_set_tv( const t_tensor t, const uint32_t *offsets, const t_value tv);
void t_set( const t_tensor t, const uint32_t *offsets, const double v);

// if with_header == TRUE, dump type, rank, shape and strides
void t_dump( const t_tensor t, const uint8_t with_header, const uint8_t verbose);

void t_dump_head( const t_tensor t, const int32_t n_rows);

// frees allocated memory. no-op if t == NULL
void t_free( t_tensor t);

#define T_FREE( t) { t_free( t); t = NULL; }

// shape can not contain negative values a la numpy.reshape()...
t_tensor t_reshape( const t_tensor t, const uint16_t rank, const t_shape shape, t_tensor out);

t_tensor t_ravel( const t_tensor t, const uint8_t copy);

void t_get_shape( const t_tensor t, t_shape_struct *shape);

// for rank == 0, 1, or 2.
// rank == 0: no-op
// rank == 1: column vector into row vector and vice-versa
// rank == 2: matrix transpose
t_tensor t_transpose( const t_tensor t, const t_tensor out);

// mimicking the numpy.transpose...
// if axes_len == 0 && axes == NULL: all axes are reversed
// axes is an array of uint8_t, thus >= 0, unlike in the numpy.transpose() where
// axes-indices can be < 0
// here axes[i] indicates a positive axis-index, and *not* a mere boolean...
// returns a newly allocated tensor
t_tensor t_transpose_axes( const t_tensor t, uint32_t axes_len, t_axes axes);

t_tensor t_apply( const t_tensor t, double (*func) (double), t_tensor out);

uint8_t t_broadcastable( const t_tensor a, const t_tensor b);

t_tensor t_broadcast_to( const t_tensor a,
			 const uint16_t rank,
			 const t_shape shape);

t_tensor t_dot( t_tensor a, t_tensor b, t_tensor out);

// compute the outer product of two vectors.
// for vectors of lengths n, m return matrix (n x m)
t_tensor t_outer( t_tensor a, t_tensor b, t_tensor out);

t_tensor t_inner( t_tensor a, t_tensor b, t_tensor out);

// element-wise operations of tensors
t_tensor t_multiply( const t_tensor a, const t_tensor b, t_tensor out);
t_tensor t_add( const t_tensor a, const t_tensor b, t_tensor out);
t_tensor t_subtract( const t_tensor a, const t_tensor b, t_tensor out);
t_tensor t_divide( const t_tensor a, const t_tensor b, t_tensor out);
t_tensor t_power( const t_tensor a, const t_tensor b, t_tensor out);

t_tensor t_matmul( const t_tensor a, const t_tensor b, t_tensor out);

t_tensor t_one();
t_tensor t_zero();
t_tensor t_minus_one();

#define T_ROW_AXIS 0  // vertical axis
#define T_COL_AXIS 1  // horizontal axis
#define T_DEPTH_AXIS 2  // depth axis

// length of axis == rank of tensor
// if axes[i] == 1, operation is applied along that axis
// axes[i] == (0|1): this differs from numpy where axes are indices, not an array of boolean
// axis 0: row axis, vertically down T_ROW_AXIS , i.e. over column
// axis 1: column axis, horizontally across T_COL_AXIS, i.e. over row
// axis 2: depth, across x-y planes T_DEPTH_AXIS

// if axis == NULL or all 1s returns a 0D tensor
t_tensor t_max( const t_tensor a, const t_axes axes, const uint8_t keep_dims);
t_tensor t_min( const t_tensor a, const t_axes axes, const uint8_t keep_dims);
t_tensor t_sum( const t_tensor a, const t_axes axes, const uint8_t keep_dims);

t_tensor t_abs( const t_tensor a, const t_tensor out);
t_tensor t_log( const t_tensor a, const t_tensor out);
t_tensor t_exp( const t_tensor a, const t_tensor out);
t_tensor t_sign( const t_tensor a, const t_tensor out);
t_tensor t_negate( const t_tensor a, const t_tensor out);

t_tensor t_relu( const t_tensor a, const t_tensor out);
t_tensor t_relu_deriv( const t_tensor a, const t_tensor out);

t_tensor t_sigmoid( const t_tensor a, const t_tensor out);
t_tensor t_sigmoid_deriv( const t_tensor a, const t_tensor out);

// compute numerically stable LogSoftMax row-wise for a 2D matrix t.
// returns a 2D tensor of same shape as t.
t_tensor t_log_softmax( const t_tensor t, const t_tensor out);

// compute numerically stable LogSumExp row-wise for a 2D matrix X
// returns a newly allocated 1D row vector of length == nbr of rows of X.
t_tensor t_log_sumexp( const t_tensor x, const t_tensor out);

t_tensor t_tanh( const t_tensor a, const t_tensor out);
t_tensor t_tanh_deriv( const t_tensor a, const t_tensor out);

t_tensor t_squeeze( t_tensor a, const t_axes axes);

t_tensor t_diag( const t_tensor v);
t_tensor t_tile( const t_tensor t, const uint16_t len_reps, const uint16_t reps[]);

uint8_t t_is_square( const t_tensor t);
t_tensor t_diagonal( const uint16_t rank, const t_shape shape, const uint8_t dtype);
t_tensor t_derivative( const t_tensor t);

// division by scalar. if !in_situ, returns a newly allocated tensor
t_tensor t_div_scalar( const t_tensor t, const double denom, const uint8_t in_situ);
// multiplication by scalar. if !in_situ, returns a newly allocated tensor
t_tensor t_mul_scalar( const t_tensor t, const double m, const uint8_t in_situ);
t_tensor t_pow_scalar( const t_tensor t, const double m, const uint8_t in_situ);
t_tensor t_add_scalar( const t_tensor t, const double m, const uint8_t in_situ);

t_tensor t_concatenate( const uint16_t len_t, const t_tensor t[], const uint8_t axis);

void t_exchange_lowest_dim( const t_tensor t,
			    const uint32_t src_idx,
			    const uint32_t tgt_idx,
			    uint8_t **bufr);

// index is index into tensor. must be of length <= t->rank. if len_index < t->rank, the high
// dimension values are set to 0 and we extract the corresponding sub-tensor.
t_tensor t_extract( const t_tensor t, const uint32_t index[], const uint16_t len_index,
		    t_tensor out, const uint8_t dtype);

// returns a tensor shortened along row-axis (axes[0]); data is copied either from head or tail.
t_tensor t_head( const t_tensor t, const uint32_t n_rows);
t_tensor t_tail( const t_tensor t, const uint32_t n_rows);

// if axis == -1: entire tensor, one index as a result index[0][rank]
// if axis >= 0 && < rank: index must be of dimension shape[axis] i.e. index[shape[axis]][rank]
// len_index either 1 or shape[axis]
void t_argmax( const t_tensor t, const int8_t axis, uint32_t *index, const uint32_t len_index);

// returns a permutation of [0..n-1], newly allocated tensor
t_tensor t_random_permutation( const uint32_t n, const uint8_t dtype);

// returns a shuffled index over dominant axis of t. no duplicates.
// newly allocated tensor.
t_tensor t_shuffled_index( const t_tensor t);

void t_write( const t_tensor t, FILE *f);
t_tensor t_read( FILE *f);

#define T_ORD_FROBENIUS 1
double t_norm( const t_tensor t, const uint8_t ord);

void dump_shape( const uint16_t rank, const t_shape shape);

int64_t prod_of_arr( const uint32_t start, const uint32_t end, const uint32_t *arr);
#endif // __TENSOR_H__
