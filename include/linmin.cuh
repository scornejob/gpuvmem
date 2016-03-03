#ifndef LINMIN_CUH
#define LINMIN_CUH

#include "mnbrak.cuh"
#include "brent.cuh"
#include "f1dim.cuh"


__host__ void linmin(cufftComplex *p, float *xi, float *fret, float (*func)(cufftComplex*));
#endif
