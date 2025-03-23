#include "mem.h"
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <limits.h>
#include <float.h>
#include <stdarg.h>

#if 0

long long allocated_bytes = 0;


void *mem_alloc( unsigned long nbr_bytes, const char *fn, const unsigned long line) {

#ifdef MEM_TRACE

  nbr_bytes += sizeof( char *) + 2 * sizeof( unsigned long);
  allocated_bytes += nbr_bytes;
  
  fprintf( stderr, "mem_alloc: %s %ld %ld %lld\n", fn, line, nbr_bytes, allocated_bytes);

  char *p = calloc( 1, nbr_bytes);
  char **pp = (char **) p;
  *pp = (char *) fn;
  
  p += sizeof( char *);
  
  unsigned long *lp = (unsigned long *) p;
  *lp = line;
  p += sizeof( unsigned long);
  
  *(lp+1) = nbr_bytes;
  p += sizeof( unsigned long);
  
  return p;

#else
  return malloc( nbr_bytes);
#endif
}

void *mem_calloc( const unsigned long cnt, const unsigned long sz, const char *fn, const unsigned int line) {
#ifdef MEM_TRACE
  void *p = mem_alloc( cnt * sz, fn, line);
  // memset( p, cnt*sz, 0);
  return p;
#else
  return calloc( cnt, sz);
#endif
}

void mem_free( const void *p) {

#ifdef MEM_TRACE
  p -= sizeof( unsigned long);
  const unsigned long *lp = (unsigned long *) p;
  const unsigned long nbr_bytes = *lp;
  
  p -= sizeof( unsigned long);

  p -= sizeof( char *);
  char **cp = (char **) p;
  
  allocated_bytes -= nbr_bytes;

  fprintf( stderr, "mem_free: %s %ld %ld %lld\n", *cp, *(lp-1), nbr_bytes, allocated_bytes);
#endif
  
  free( (void *) p);
  
}

#endif


// to chase memory leaks...
#ifdef TRACE_MEM

typedef struct {
  void *a;
  uint32_t len;
  uint32_t size;
  char *file;
  uint32_t line;
} mem_calloc_struct;

#define MEM_TBL_SZ 25000

static mem_calloc_struct mem_tbl[MEM_TBL_SZ];
static uint32_t mem_initialized = FALSE;
static long long mem_count = 0;

static void init_mem() {
  if ( mem_initialized)
    return;
  memset( mem_tbl, 0, sizeof( mem_tbl));
  mem_initialized = TRUE;
}

static void enter_calloc( const uint32_t len, const uint32_t size, const char *file, const uint32_t line,
		     const void *addr) {

  init_mem();

  // fprintf( stderr, "calloc: %s %d: %d %d %lx\n", file, line, len, size, (unsigned long) addr);

  mem_calloc_struct *p = mem_tbl;
  for ( uint32_t i = 0; i < MEM_TBL_SZ; i++) {
    // find a free slot...
    if ( p->a == NULL) {

      p->a = (void *) addr;
      p->len = len;
      p->size = size;
      p->file = (char *) file;
      p->line = line;

      mem_count += len*size;

#if 0
      {
	unsigned char buf[1024]; memset( buf, 0, sizeof( buf));
	sprintf( buf, "e %d\n", i);
	sprintf( buf+strlen( buf), "calloc: %s %d: %d %d %lx\n", file, line, len, size, (unsigned long) addr);
	fprintf( stderr, "%s", buf);
      }
#endif
      
      return;
    }
    p++;
  }

  fprintf( stderr, "mem tbl full...\n");
  mem_dump_tbl();
  exit( -1);
  
}

static void delete_calloc( const void *addr, const char *file, const uint32_t line) {

  // fprintf( stderr, "free: %s %d %lx\n", file, line, (unsigned long) addr);

  mem_calloc_struct *p = mem_tbl;
  for ( uint32_t i = 0; i < MEM_TBL_SZ; i++) {
    // search the matching addr entry
    if ( p->a == addr) {

#if 0      
      {
	unsigned char buf[1024]; memset( buf, 0, sizeof( buf));	
	sprintf( buf, "d %d\n", i);
	sprintf( buf+strlen( buf), "free: %s %d %lx\n", file, line, (unsigned long) addr);
	fprintf( stderr, "%s", buf);
      }
#endif
      p->a = NULL;
      mem_count -= p->len*p->size;
      memset( p, 0, sizeof( mem_calloc_struct));
      
      return;
    }
    p++;
  }
}

void mem_dump_tbl() {
  fprintf( stderr, "dumping mem tbl: mem_count = %lld\n", mem_count);
  
  mem_calloc_struct *p = mem_tbl;
  for ( uint32_t i = 0; i < MEM_TBL_SZ; i++) {
    if ( p->a != NULL) {
      fprintf( stderr, "%s %d %d %d %lx\n", p->file, p->line, p->len, p->size, (long unsigned int) p->a);
    }
    p++;
  }
}

void *mem_calloc( const uint32_t len, const uint32_t size, const char *file, const uint32_t line) {
  void *m = calloc( len, size);
  enter_calloc( len, size, file, line, m);
  return m;
}

void mem_free( void *ptr, const char *file, const uint32_t line) {
  delete_calloc( ptr, file, line);
  free( ptr);
}
#endif // TRACE_MEM



