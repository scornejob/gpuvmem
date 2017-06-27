#ifndef FRPRMN_CUH
#define FRPRMN_CUH
#include "linmin.cuh"

__host__ void frprmn(float3 *p, float ftol, float *fret, float (*func)(float3*), void (*dfunc)(float3*, float3*), int flag);
#endif
