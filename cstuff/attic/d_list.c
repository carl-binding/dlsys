#include "d_list.h"

#include <stdlib.h>
#include <assert.h>
#include <string.h>

d_list dl_new( const unsigned int sz, const unsigned char type) {
  d_list dl = (d_list) calloc( 1, sizeof(d_list_struct));
  dl->sz = sz;
  dl->type = type;
  dl->data = (d_list_el *) calloc( sz, sizeof( d_list_el));
  return dl;
}

void dl_reset( d_list dl) {
  if ( dl == NULL)
    return;

  if ( dl->type == T_PTR) {
    for ( unsigned int i = 0 ; i < dl->cnt; i++) {
      free( dl->data[i].ptr);
    }
  }
  dl->cnt = 0;
}

void dl_free( d_list dl) {
  if ( dl == NULL)
    return;
  
  if ( dl->type == T_PTR) {
    for ( unsigned int i = 0 ; i < dl->cnt; i++) {
      free( dl->data[i].ptr);
    }
  }
  free( dl->data);
  free( dl);
}

void dl_append( const d_list dl, d_list_el v) {
  if ( dl->cnt >= dl->sz) {
    const unsigned int n_sz = dl->sz*2;
    d_list_el *dd = (d_list_el *) calloc( n_sz, sizeof( d_list_el));
    memcpy( dd, dl->data, dl->sz*sizeof( d_list_el));
    free( dl->data);
    dl->data = dd;
    dl->sz = n_sz;
  }
  assert( dl->cnt < dl->sz);
  dl->data[dl->cnt++] = v;
}

void dl_set( const d_list dl, const unsigned int i, d_list_el v) {
  assert( i < dl->cnt);
  dl->data[i] = v;
}

d_list_el dl_get( const d_list dl, const unsigned int i) {
  assert( i < dl->cnt);
  return dl->data[i];
}


double dl_sum( const d_list dl) {

  if ( dl->type == T_PTR) {
    assert( 0);
    return 0;
  }
  
  double s = 0.0;
  for ( unsigned int i = 0; i < dl->cnt; i++) {
    d_list_el *e = &(dl->data[i]);
    switch ( dl->type) {
    case T_INT8:
      s += (double) e->c;
      break;
    case T_INT16:
      s += (double) e->s;
      break;
    case T_INT32:
      s += (double) e->l;
      break;
    case T_INT64:
      s += (double) e->ll;
      break;
    case T_FLOAT:
      s += (double) e->f;
      break;
    case T_DOUBLE:
      s += (double) e->d;
      break;
    default:
      assert( 0);
    }    

  }
  return s;
}
