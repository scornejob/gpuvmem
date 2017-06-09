#ifndef LINMIN_CUH
#define LINMIN_CUH

#include "mnbrak.cuh"
#include "brent.cuh"
#include "f1dim.cuh"


__host__ void linmin(float2 *p, float2 *xi, float *fret, float (*func)(float2*));
#endif
