#ifndef __TUPLE_H__
#define __TUPLE_H__

#include <stdarg.h>
#include <stdio.h>

#include "list.h" // for defn of supported data types

typedef struct {
  unsigned int sz;
  unsigned char dtype;
  l_el *data;
} tuple_struct, *tuple;

tuple tpl_new( unsigned char dtype,
	       unsigned int sz,
	       ...);

void tpl_free( tuple tpl);

unsigned int tpl_len( const tuple tpl);
l_el tpl_get( const tuple tpl, unsigned int idx);

#endif
