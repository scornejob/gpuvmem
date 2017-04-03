#ifndef LINMIN_CUH
#define LINMIN_CUH

#include "mnbrak.cuh"
#include "brent.cuh"
#include "f1dim.cuh"


__host__ void linmin(float3 *p, float3 *xi, float *fret, float (*func)(float3*));
#endif
