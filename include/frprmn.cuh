#ifndef FRPRMN_CUH
#define FRPRMN_CUH
#include "linmin.cuh"

__host__ void armijoTest(cufftComplex *p, float (*func)(cufftComplex*), void (*dfunc)(cufftComplex*, float*));
__host__ void frprmn(cufftComplex *p, float ftol, float *fret, float (*func)(cufftComplex*), void (*dfunc)(cufftComplex*, float*));
#endif
