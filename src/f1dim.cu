#include "f1dim.cuh"
extern cufftComplex *device_pcom;
extern float *device_xicom, (*nrfunc)(cufftComplex*);
extern long M;
extern long N;
extern float MINPIX;
extern dim3 threadsPerBlockNN;
extern dim3 numBlocksNN;
extern int nopositivity;

__host__ float f1dim(float x)
{
    cufftComplex *device_xt;
    float f;

    gpuErrchk(cudaMalloc((void**)&device_xt, sizeof(cufftComplex)*M*N));
    gpuErrchk(cudaMemset(device_xt, 0, sizeof(cufftComplex)*M*N));

    //printf("Se evalua en f1dim %f\n", x);
    //xt = pcom+x*xicom;
    if(nopositivity == 0){
      evaluateXt<<<numBlocksNN, threadsPerBlockNN>>>(device_xt, device_pcom, device_xicom, x, MINPIX, N);
      gpuErrchk(cudaDeviceSynchronize());
    }else{
      evaluateXtNoPositivity<<<numBlocksNN, threadsPerBlockNN>>>(device_xt, device_pcom, device_xicom, x, N);
      gpuErrchk(cudaDeviceSynchronize());
    }

    f = (*nrfunc)(device_xt);
    cudaFree(device_xt);
    return f;
}
