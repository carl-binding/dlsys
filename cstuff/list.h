#ifndef _LIST_H_
#define _LIST_H_

// a multi-typed list using array of data

// supported data types
#define T_INT8  1
#define T_INT16 3
#define T_INT32 5
#define T_INT64 7
#define T_FLOAT 9
#define T_DOUBLE 11
#define T_PTR 13

typedef union {
  char c;
  short s;
  long l;        // 32 bits
  long long ll;  // 64 bits
  float f;
  double d;
  void *ptr;
} l_el;

typedef struct {
  unsigned char type;
  unsigned int sz;
  unsigned int cnt;
  l_el *data;
  void (*free_func) (void *); // NULL for non-ptr types
} l_list_struct, *l_list;

#define l_len( l) (l==NULL?0:l->cnt)


// NULL for type != T_PTR. if NULL when type == T_PTR individual
// elements are not freed. Needed in case we have shared references.
l_list l_new( unsigned int sz, const unsigned char type,
	      void (*free_func) (void *));
void l_free( l_list l);

// clears list, freeing elements if free_func() was set.
void l_reset( const l_list l);

void l_append( const l_list l, l_el el);
void l_append_unique( const l_list l, l_el el);

uint8_t l_contains( const l_list l, l_el v);

// convenience
void l_append_ptr( const l_list l, const void *p);

void l_set( const l_list l, const unsigned int i, l_el el);
l_el l_get( const l_list l, const unsigned int i);

void l_set_p( const l_list l, const unsigned int i, const void *p);
void *l_get_p( const l_list l, const unsigned int i);

double l_sum( const l_list l);

l_list l_reverse( const l_list l);

void l_dump( const l_list l);

// returns a (shallow) copy of list l
l_list l_copy( const l_list l);

l_list l_delete( const l_list l, l_el el);

#endif
