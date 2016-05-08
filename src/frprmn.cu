#include "frprmn.cuh"

extern long M;
extern long N;
extern int iter;

extern dim3 threadsPerBlockNN;
extern dim3 numBlocksNN;

extern int threadsVectorNN;
extern int blocksVectorNN;

extern float MINPIX;

#define EPS 1.0e-10
#define ITERATIONS 500

#define ARMIJOTOLERANCE 1.e-6
#define ALPHA 1.e-4


#define FREEALL cudaFree(device_gg_vector);cudaFree(device_dgg_vector);cudaFree(xi);cudaFree(device_h);cudaFree(device_g);



__host__ void armijoTest(cufftComplex *p, float (*func)(cufftComplex*), void (*dfunc)(cufftComplex*, float*))
{
  int i = 0;
  int iarm;
  float normPGC, normPL, fc, ft, fgoal;
  float *device_normVector, *device_pgc, *device_x, *xi, *device_xt, *device_pl;

  gpuErrchk(cudaMalloc((void**)&device_normVector, sizeof(float)*M*N));
  gpuErrchk(cudaMemset(device_normVector, 0, sizeof(float)*M*N));

  gpuErrchk(cudaMalloc((void**)&device_pgc, sizeof(float)*M*N));
  gpuErrchk(cudaMemset(device_pgc, 0, sizeof(float)*M*N));

  gpuErrchk(cudaMalloc((void**)&device_x, sizeof(float)*M*N));
  gpuErrchk(cudaMemset(device_x, 0, sizeof(float)*M*N));

  gpuErrchk(cudaMalloc((void**)&xi, sizeof(float)*M*N));
  gpuErrchk(cudaMemset(xi, 0, sizeof(float)*M*N));

  gpuErrchk(cudaMalloc((void**)&device_xt, sizeof(float)*M*N));
  gpuErrchk(cudaMemset(device_xt, 0, sizeof(float)*M*N));

  gpuErrchk(cudaMalloc((void**)&device_pl, sizeof(float)*M*N));
  gpuErrchk(cudaMemset(device_pl, 0, sizeof(float)*M*N));


  fc = (*func)(p);
  printf("Function value = %f\n", fc);
  //exit(0);
  (*dfunc)(p,xi);

  substraction<<<numBlocksNN, threadsPerBlockNN>>>(device_x, p, xi, 1.0, N);
  gpuErrchk(cudaDeviceSynchronize());

  projection<<<numBlocksNN, threadsPerBlockNN>>>(device_xt, device_x, MINPIX, N);
  gpuErrchk(cudaDeviceSynchronize());

  substraction<<<numBlocksNN, threadsPerBlockNN>>>(device_pgc, p, device_xt, 1.0, N);
  gpuErrchk(cudaDeviceSynchronize());

  normVectorCalculation<<<numBlocksNN, threadsPerBlockNN>>>(device_normVector, device_pgc, N);
  gpuErrchk(cudaDeviceSynchronize());

  normPGC = deviceReduce(device_normVector, M*N);
  i=1;
  while(normPGC > ARMIJOTOLERANCE && i <= ITERATIONS){
    iter = i;
    float lambda2 = 1.0;

    substraction<<<numBlocksNN, threadsPerBlockNN>>>(device_x, p, xi, lambda2, N);
    gpuErrchk(cudaDeviceSynchronize());

    projection<<<numBlocksNN, threadsPerBlockNN>>>(device_xt, device_x, MINPIX, N);
    gpuErrchk(cudaDeviceSynchronize());

    ft = (*func)(p);

    substraction<<<numBlocksNN, threadsPerBlockNN>>>(device_pl, p, xi, 1.0, N);
    gpuErrchk(cudaDeviceSynchronize());

    normVectorCalculation<<<numBlocksNN, threadsPerBlockNN>>>(device_normVector, device_pl, N);
    gpuErrchk(cudaDeviceSynchronize());

    normPL = deviceReduce(device_normVector, M*N);

    fgoal = fc * normPL *(ALPHA/lambda2);
    iarm = 0;
    while(ft<fgoal){
      lambda2 = lambda2 * 0.1;
      substraction<<<numBlocksNN, threadsPerBlockNN>>>(device_x, p, xi, lambda2, N);
      gpuErrchk(cudaDeviceSynchronize());

      projection<<<numBlocksNN, threadsPerBlockNN>>>(device_xt, device_x, MINPIX, N);
      gpuErrchk(cudaDeviceSynchronize());

      substraction<<<numBlocksNN, threadsPerBlockNN>>>(device_pl, p, device_xt, lambda2, N);
      gpuErrchk(cudaDeviceSynchronize());

      ft = (*func)(p);

      normVectorCalculation<<<numBlocksNN, threadsPerBlockNN>>>(device_normVector, device_pl, N);
      gpuErrchk(cudaDeviceSynchronize());

      normPL = deviceReduce(device_normVector, M*N);
      iarm++;
      if(iarm>10){
        break;
      }
      fgoal = fc * normPL * (ALPHA/lambda2);
    }

    //p.x = xt
    copyImage<<<numBlocksNN, threadsPerBlockNN>>>(p, device_xt, N);
    gpuErrchk(cudaDeviceSynchronize());

    fc = (*func)(p);
    printf("Function value = %f\n", fc);
    //exit(0);
    (*dfunc)(p,xi);

    substraction<<<numBlocksNN, threadsPerBlockNN>>>(device_x, p, xi, 1.0, N);
    gpuErrchk(cudaDeviceSynchronize());

    projection<<<numBlocksNN, threadsPerBlockNN>>>(device_pgc, device_x, MINPIX, N);
    gpuErrchk(cudaDeviceSynchronize());

    normVectorCalculation<<<numBlocksNN, threadsPerBlockNN>>>(device_normVector, device_pgc, N);
    gpuErrchk(cudaDeviceSynchronize());

    normPGC = deviceReduce(device_normVector, M*N);
    i++;
  }

  cudaFree(device_normVector);
  cudaFree(device_pgc);
  cudaFree(device_x);
  cudaFree(xi);
  cudaFree(device_xt);
  cudaFree(device_pl);

}


__host__ void frprmn(cufftComplex *p, float ftol, float *fret, float (*func)(cufftComplex*), void (*dfunc)(cufftComplex*, float*))
{
  float gg, dgg, gam, fp;
  float *device_g, *device_h, *xi;


  //////////////////////MEMORY GPU//////////////////////////
  gpuErrchk(cudaMalloc((void**)&device_g, sizeof(float)*M*N));
  gpuErrchk(cudaMemset(device_g, 0, sizeof(float)*M*N));
  gpuErrchk(cudaMalloc((void**)&device_h, sizeof(float)*M*N));
  gpuErrchk(cudaMemset(device_h, 0, sizeof(float)*M*N));
  gpuErrchk(cudaMalloc((void**)&xi, sizeof(float)*M*N));
  gpuErrchk(cudaMemset(xi, 0, sizeof(float)*M*N));

  ///////////////////vectors for gg and dgg////////////////////
  float *device_gg_vector, *device_dgg_vector;

  //////////////////////////////////GPU MEMORY///////////////
  gpuErrchk(cudaMalloc((void**)&device_gg_vector, sizeof(float)*M*N));
  gpuErrchk(cudaMemset(device_gg_vector, 0, sizeof(float)*M*N));

  gpuErrchk(cudaMalloc((void**)&device_dgg_vector, sizeof(float)*M*N));
  gpuErrchk(cudaMemset(device_dgg_vector, 0, sizeof(float)*M*N));

  fp = (*func)(p);
  printf("Function value = %f\n", fp);
  (*dfunc)(p,xi);
  //g=-xi
  //xi=h=g
  searchDirection<<<numBlocksNN, threadsPerBlockNN>>>(device_g, xi, device_h, N);//Search direction
  gpuErrchk(cudaDeviceSynchronize());



  ////////////////////////////////////////////////////////////////
  for(int i=1; i <= ITERATIONS; i++){
    iter = i;
    printf("\n\n**********Iteration %d **********\n\n", i);
    linmin(p, xi, fret, func);

    if (2.0*fabs(*fret-fp) <= ftol*(fabs(*fret)+fabs(fp)+EPS)) {
      printf("Exit due to tolerance\n");
      FREEALL
			return;
		}

    fp=(*func)(p);
    printf("Function value = %f\n", fp);
    (*dfunc)(p,xi);
    dgg = gg = 0.0;
    ////gg = g*g
    ////dgg = (xi+g)*xi
    getGandDGG<<<numBlocksNN, threadsPerBlockNN>>>(device_gg_vector, device_dgg_vector, xi, device_g, N);
  	gpuErrchk(cudaDeviceSynchronize());
    ////getSums (Reductions) of gg dgg
    gg = deviceReduce(device_gg_vector, M*N);
    dgg = deviceReduce(device_dgg_vector, M*N);
    if(gg == 0.0){
      printf("Exit due to gg = 0\n");
      FREEALL
      return;
    }
    gam = dgg/gg;
    //printf("Gamma = %f\n", gam);
    //g=-xi
    //xi=h=g+gam*h;
    newXi<<<numBlocksNN, threadsPerBlockNN>>>(device_g, xi, device_h, gam, N);
  	gpuErrchk(cudaDeviceSynchronize());

  }
  printf("Too many iterations in frprmn\n");
  FREEALL
  return;
}
