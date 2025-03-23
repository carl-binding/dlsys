#include <stdlib.h>
#include <assert.h>

#include "mem.h"
#include "tuple.h"

tuple tpl_new( unsigned char dtype,
	       unsigned int sz,
	       ...) {

  tuple t = (tuple) MEM_CALLOC( 1, sizeof( tuple_struct));
  t->data = (l_el *) MEM_CALLOC( sz, sizeof( l_el));
  t->dtype = dtype;
  t->sz = sz;

  va_list args;
  va_start(args, sz);  

  for (int i = 0; i < sz; i++) {
    l_el el;

    switch ( dtype) {
      case T_INT8:
	el.c = (char) va_arg( args, int);
	break;
      case T_INT16:
	el.s = (short) va_arg( args, int);
	break;
      case T_INT32:
	el.l = va_arg( args, long);
	break;
      case T_INT64:
	el.ll = va_arg( args, long long);
	break;
      case T_FLOAT:
	el.f = (float) va_arg( args, double);
	break;
      case T_DOUBLE:
	el.d = va_arg( args, double);
	break;
      case T_PTR:
	el.ptr = va_arg( args, void *);
	break;
      default:
        assert( 0);
    }
    t->data[i] = el;
  }

  va_end( args);
  
  return t;
}

void tpl_free( tuple tpl) {
  if ( tpl == NULL)
    return;
  MEM_FREE( tpl->data);
  MEM_FREE( tpl);
}

unsigned int tpl_len( const tuple tpl) {
  return tpl->sz;
}

l_el tpl_get( const tuple tpl, unsigned int idx) {
  assert( tpl != NULL);
  if ( idx >= tpl->sz) {
    fprintf( stderr, "tpl_get: index out of range %d %d\n", idx, tpl->sz);
    exit( -1);
  }
  return tpl->data[idx];
}
