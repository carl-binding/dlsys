#ifndef __LOGGER_H__
#define __LOGGER_H__

#include <stdarg.h>

#define LOG_EMERGENCY 0
#define LOG_ALERT 1
#define LOG_CRITICAL 2
#define LOG_ERROR 3
#define LOG_WARNING 4
#define LOG_NOTICE 5
#define LOG_INFO 6
#define LOG_TRACE LOG_INFO
#define LOG_DEBUG 7

void log_set_level( int _level);
int log_get_level();

void log_msg( const int _level, const char *fmt, ...);

#endif
