#include "linmin.cuh"
#define TOL 1.0e-7

cufftComplex *device_pcom;
float *device_xicom, (*nrfunc)(cufftComplex*);
extern long M;
extern long N;
extern float MINPIX;
extern int positivity;

extern dim3 threadsPerBlockNN;
extern dim3 numBlocksNN;
extern int verbose_flag;


__host__ void linmin(cufftComplex *p, float *xi, float *fret, float (*func)(cufftComplex*))//p and xi are in GPU
{
  float xx, xmin, fx, fb, fa, bx ,ax;

  gpuErrchk(cudaMalloc((void**)&device_pcom, sizeof(cufftComplex)*M*N));
  gpuErrchk(cudaMemset(device_pcom, 0, sizeof(cufftComplex)*M*N));

  gpuErrchk((cudaMalloc((void**)&device_xicom, sizeof(float)*M*N)));
  gpuErrchk(cudaMemset(device_xicom, 0, sizeof(float)*M*N));
  nrfunc = func;
  //device_pcom = p;
  //device_xicom = xi;
  gpuErrchk(cudaMemcpy2D(device_pcom, sizeof(cufftComplex), p, sizeof(cufftComplex), sizeof(cufftComplex), M*N, cudaMemcpyDeviceToDevice));
  gpuErrchk(cudaMemcpy2D(device_xicom, sizeof(float), xi, sizeof(float), sizeof(float), M*N, cudaMemcpyDeviceToDevice));

  ax = 0.0;
	xx = 1.0;

  mnbrak(&ax, &xx, &bx, &fa, &fx, &fb, f1dim);


  *fret = brent(ax, xx, bx, TOL, &xmin, f1dim);
  if(verbose_flag){
    printf("alpha = %f\n\n", xmin);
  }

  //GPU MUL AND ADD
  //xi     = xi*xmin;
  //p      = p + xi;
  if(positivity == 1){
    newP<<<numBlocksNN, threadsPerBlockNN>>>(p, xi, xmin, MINPIX, N);
    gpuErrchk(cudaDeviceSynchronize());
  }else{
    newPNoPositivity<<<numBlocksNN, threadsPerBlockNN>>>(p, xi, xmin, N);
    gpuErrchk(cudaDeviceSynchronize());
  }

  cudaFree(device_xicom);
  cudaFree(device_pcom);
}
#undef TOL
