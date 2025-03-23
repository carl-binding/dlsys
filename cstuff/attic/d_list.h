#ifndef _D_LIST_H_
#define _D_LIST_H_

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
} d_list_el;

typedef struct {
  unsigned char type;
  unsigned int sz;
  unsigned int cnt;
  d_list_el *data;
} d_list_struct, *d_list;

d_list dl_new( const unsigned int sz, const unsigned char type);
void dl_free( d_list dl);
void dl_reset( d_list dl);

void dl_append( const d_list dl, d_list_el el);

void dl_set( const d_list dl, const unsigned int i, d_list_el el);
d_list_el dl_get( const d_list dl, const unsigned int i);

double dl_sum( const d_list dl);

#endif
