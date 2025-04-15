#include "mem.h"
#include "list.h"

#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <stdio.h>

l_list l_new( unsigned int sz,
	      const unsigned char type,
	      void (*free_func)(void *)) {

  if ( sz == 0)
    sz = 1;

  if ( type != T_PTR)
    assert( free_func == NULL);
  
  l_list l = (l_list) MEM_CALLOC( 1, sizeof(l_list_struct));
  l->sz = sz;
  l->cnt = 0;
  l->type = type;
  l->data = (l_el *) MEM_CALLOC( sz, sizeof( l_el));
  l->free_func = free_func;
  return l;
}

// remove all the elements of list.
// referenced elements are also freed when a free() function has been set.
void l_reset( const l_list l) {
  if ( l == NULL)
    return;

  if ( l->type == T_PTR) {
    if ( l->free_func != NULL) {
      for ( unsigned int i = 0 ; i < l->cnt; i++) {
	(*l->free_func)( (void *) l->data[i].ptr);
      }
    }
  }
  l->cnt = 0;
}

void l_free( l_list l) {
  if ( l == NULL)
    return;
  
  if ( l->type == T_PTR) {
    if ( l->free_func != NULL) {
      for ( unsigned int i = 0 ; i < l->cnt; i++) {
	(*l->free_func)( (void *) l->data[i].ptr);
      }
    }
  }
  MEM_FREE( l->data);
  MEM_FREE( l);
}


uint8_t l_contains( const l_list l, l_el v) {
  if ( l == NULL)
    return FALSE;

  for ( int i = 0; i < l->cnt; i++) {
    l_el el = l_get( l, i);
    // we can't use memcmp() here since we don't know what l_el
    // contains in cases of non-aligned data...
    switch ( l->type) {
      case T_INT8:
	if ( v.c == el.c)
	  return TRUE;
	break;
      case T_INT16:
	if ( v.s == el.s)
	  return TRUE;
	break;
      case T_INT32:
	if ( v.l == el.l)
	  return TRUE;
	break;
      case T_INT64:
	if ( v.ll == el.ll)
	  return TRUE;
	break;
      case T_FLOAT:
	if ( v.f == el.f)
	  return TRUE;
	break;
      case T_DOUBLE:
	if ( v.d == el.d)
	  return TRUE;
	break;
      case T_PTR:
	if ( v.ptr == el.ptr)
	  return TRUE;
	break;
      default:
        assert( 0);
    }
  }
  return FALSE;  
}


// this is *not* efficient... but for short lists, who cares?
void l_append_unique( const l_list l, l_el v) {
  for ( int i = 0; i < l->cnt; i++) {
    l_el el = l_get( l, i);
    // we can't use memcmp() here since we don't know what l_el
    // contains in cases of non-aligned data...
    switch ( l->type) {
      case T_INT8:
	if ( v.c == el.c)
	  return;
	break;
      case T_INT16:
	if ( v.s == el.s)
	  return;
	break;
      case T_INT32:
	if ( v.l == el.l)
	  return;
	break;
      case T_INT64:
	if ( v.ll == el.ll)
	  return;
	break;
      case T_FLOAT:
	if ( v.f == el.f)
	  return;
	break;
      case T_DOUBLE:
	if ( v.d == el.d)
	  return;
	break;
      case T_PTR:
	if ( v.ptr == el.ptr)
	  return;
	break;
      default:
        assert( 0);
    }
  }
  l_append( l, v);
}

void l_append_ptr( const l_list l, const void *p) {
  assert( l->type == T_PTR);
  l_el e;
  e.ptr = (void *) p;
  l_append( l, e);
}

void l_append( const l_list l, l_el v) {
  if ( l->cnt >= l->sz) {
    const unsigned int n_sz = l->sz*2;
    l_el *dd = (l_el *) MEM_CALLOC( n_sz, sizeof( l_el));
    memcpy( dd, l->data, l->sz*sizeof( l_el));
    MEM_FREE( l->data);
    l->data = dd;
    l->sz = n_sz;
  }
  l->data[l->cnt++] = v;
}

void l_set( const l_list l, const unsigned int i, l_el v) {
  assert( i < l->cnt);
  l->data[i] = v;
}

void l_set_p( const l_list l, const unsigned int i, const void *p) {
  assert( l->type == T_PTR);
  l_el e;
  e.ptr = (void *) p;
  l_set( l, i, e);
}

l_el l_get( const l_list l, const unsigned int i) {
  assert( i < l->cnt);
  return l->data[i];
}

void *l_get_p( const l_list l, const unsigned int i) {
  assert( l->type == T_PTR);
  assert( i < l->cnt);
  l_el el = l->data[i];
  return el.ptr;
}

double l_sum( const l_list l) {

  if ( l->type == T_PTR) {
    assert( 0);
    return 0;
  }
  
  double s = 0.0;
  for ( unsigned int i = 0; i < l->cnt; i++) {
    l_el *e = &(l->data[i]);
    switch ( l->type) {
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

l_list l_reverse( const l_list l) {
  l_list rl = l_new( l->sz, l->type, NULL);
  for ( int i = 0; i < l->cnt; i++) {
    l_append( rl, l_get( l, l->cnt-1-i));
  }
  return rl;
}

void l_dump( const l_list l) {
  fprintf( stderr, "[ ");
  for ( int i = 0; i < l->cnt; i++) {
    l_el *e = &(l->data[i]);
    int line_break = 0;
    switch ( l->type) {
    case T_INT8:
      fprintf( stderr, "%d ", e->c);
      line_break = 24;
      break;
    case T_INT16:
      fprintf( stderr, "%d ", e->s);
      line_break = 20;
      break;
    case T_INT32:
      fprintf( stderr, "%ld ", e->l);
      line_break = 16;
      break;
    case T_INT64:
      fprintf( stderr, "%lld ", e->ll);
      line_break = 12;
      break;
    case T_FLOAT:
      fprintf( stderr, "%f ", e->f);
      line_break = 16;      
      break;
    case T_DOUBLE:
      fprintf( stderr, "%f ", e->d);
      line_break = 16;            
      break;
    case T_PTR:
      fprintf( stderr, "0x%p ", e->ptr);
      line_break = 12;      
      break;
    default:
      assert( 0);
    }
    // comma after all but the last entry
    if ( i != (l->cnt-1))
      fprintf( stderr, ", ");
    // occasionally switch lines
    if ( (i+1) % 10 == 0)
      fprintf( stderr, "\n  ");
  }
  fprintf( stderr, "]\n");
}

l_list l_copy( const l_list l) {
  l_list ll = l_new( l->sz, l->type, NULL);
  ll->cnt = l->cnt;
  memcpy( ll->data, l->data, l->cnt * sizeof( l_el));
  return ll;
}

static int find_el( const l_list l, l_el v) {
  for ( int i = 0; i < l->cnt; i++) {
    l_el el = l_get( l, i);
    // we can't use memcmp() here since we don't know what l_el
    // contains in cases of non-aligned data...
    switch ( l->type) {
      case T_INT8:
	if ( v.c == el.c)
	  return i;
	break;
      case T_INT16:
	if ( v.s == el.s)
	  return i;
	break;
      case T_INT32:
	if ( v.l == el.l)
	  return i;
	break;
      case T_INT64:
	if ( v.ll == el.ll)
	  return i;
	break;
      case T_FLOAT:
	if ( v.f == el.f)
	  return i;
	break;
      case T_DOUBLE:
	if ( v.d == el.d)
	  return i;
	break;
      case T_PTR:
	if ( v.ptr == el.ptr)
	  return i;
	break;
      default:
	assert( 0);
    }
  }
  return -1;
}

l_list l_delete( const l_list l, l_el el) {
  int idx = find_el( l, el);

  if ( idx < 0) // not found, no-op
    return l;

  assert( l->cnt > 0);
  
  int nbr_elems_to_shift = l->cnt - (idx + 1);
  unsigned long off = idx * sizeof( l_el);
  memcpy( l->data + off, l->data + off + sizeof( l_el), nbr_elems_to_shift * sizeof( l_el));
  l->cnt -= 1;
  return l;
}


