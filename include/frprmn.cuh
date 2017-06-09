#ifndef FRPRMN_CUH
#define FRPRMN_CUH
#include "linmin.cuh"

__host__ void frprmn(float2 *p, float ftol, float *fret, float (*func)(float2*), void (*dfunc)(float2*, float2*));
#endif
