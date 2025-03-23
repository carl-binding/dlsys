#ifndef _D_LIST_H_
#define _D_LIST_H_

typedef struct {
  unsigned int sz;
  unsigned int cnt;
  double *data;
} d_list_struct, *d_list;

d_list dl_new( const unsigned int sz);
void dl_free( d_list dl);

void dl_set( const d_list dl, const unsigned int i, const double v);
double dl_get( const d_list dl, const unsigned int i);

double dl_sum( const d_list dl);

#endif
