#include <stdio.h>
#include <assert.h>


#include "logger.h"

static int level = LOG_DEBUG;
static FILE *file = NULL;

void log_set_level( int _level) {
  assert( _level >= LOG_EMERGENCY && _level <= LOG_DEBUG);
  level = _level;
}

int log_get_level() {
  return level;
}

void log_msg( const int _level, const char *fmt, ...) {
  if ( _level > level)
    return;
  va_list argptr;
  va_start(argptr,fmt);

  if ( file == NULL)
    vfprintf( stderr, fmt, argptr);
  else
    vfprintf( file, fmt, argptr);
  
  va_end(argptr);
}


