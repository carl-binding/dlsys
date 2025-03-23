#ifndef _MEM_H_
#define _MEM_H_

#include <stdint.h>

#ifndef TRUE
#define TRUE 1
#endif

#ifndef FALSE
#define FALSE 0
#endif

#if 0

void *mem_alloc( unsigned long nbr_bytes, const char *fn, const unsigned long line);
void *mem_calloc( const unsigned long cnt, const unsigned long sz, const char *fn, const unsigned int line);

#define MEM_ALLOC( nbr_bytes) mem_calloc( nbr_bytes, 1, __FILE__, __LINE__)
#define MEM_CALLOC( cnt, sz)  mem_calloc( cnt, sz, __FILE__, __LINE__)

void mem_free( const void *p);
#define MEM_FREE( p) { if ( (p) != NULL) { mem_free( (p)); (p) = NULL;}}
#endif


#define TRACE_MEM

#ifdef TRACE_MEM

void *mem_calloc( const uint32_t len, const uint32_t size, const char *file, const uint32_t line);
void mem_free( void *ptr, const char *file, const uint32_t line);

#define MEM_ALLOC( nbr_bytes) mem_calloc( nbr_bytes, 1, __FILE__, __LINE__)
#define MEM_CALLOC( len, sz)  mem_calloc( len, sz, __FILE__, __LINE__)
#define MEM_FREE( m) { if ( m != NULL) { mem_free( m, __FILE__, __LINE__); m = NULL;}}

void mem_dump_tbl();

#else // !TRACE_MEM

#define MEM_ALLOC( nbr_bytes) alloc( nbr_bytes)
#define MEM_CALLOC( len, sz) calloc( len, sz)
#define MEM_FREE( m) { if ( m != NULL) { free( m); m = NULL;}}


#endif // TRACE_MEM


#endif
