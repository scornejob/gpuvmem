#ifndef LBFGS_CUH
#define LBFGS_CUH
#include "linmin.cuh"

class LBFGS : public Optimizator
{
public:
__host__ void allocateMemoryGpu();
__host__ void deallocateMemoryGpu();
__host__ void minimizate();
private:
float ftol = 0;
float fret = 0;
float gg, dgg, gam, fp;
float *device_g, *device_h, *xi;
float *device_gg_vector, *device_dgg_vector;
int configured = 1;
float *d_s;
float *d_y, *xi, *xi_p, *norm_vector;
float norm;
float *p_p;
int K = 3;
};

#endif
