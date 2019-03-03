/* -------------------------------------------------------------------------
  Copyright (C) 2016-2017  Miguel Carcamo, Pablo Roman, Simon Casassus,
  Victor Moral, Fernando Rannou - miguel.carcamo@usach.cl

  This program includes Numerical Recipes (NR) based routines whose
  copyright is held by the NR authors. If NR routines are included,
  you are required to comply with the licensing set forth there.

	Part of the program also relies on an an ANSI C library for multi-stream
	random number generation from the related Prentice-Hall textbook
	Discrete-Event Simulation: A First Course by Steve Park and Larry Leemis,
  for more information please contact leemis@math.wm.edu

  Additionally, this program uses some NVIDIA routines whose copyright is held
  by NVIDIA end user license agreement (EULA).

  For the original parts of this code, the following license applies:

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
* -------------------------------------------------------------------------
*/

#include "frprmn.cuh"

extern long M;
extern long N;
extern int iter;

extern dim3 threadsPerBlockNN;
extern dim3 numBlocksNN;

extern int threadsVectorNN;
extern int blocksVectorNN;

extern float MINPIX;
extern int verbose_flag;

#define EPS 1.0e-10
extern int it_maximum;

#define ARMIJOTOLERANCE 1.e-6
#define ALPHA 1.e-4


#define FREEALL cudaFree(device_gg_vector);cudaFree(device_dgg_vector);cudaFree(xi);cudaFree(device_h);cudaFree(device_g);
#define FREEALL_LBFGS cudaFree(d_y);cudaFree(d_s);cudaFree(xi);cudaFree(xi_p);cudaFree(p_p);cudaFree(norm_vector);


__host__ void armijoTest(cufftComplex *p, float (*func)(cufftComplex*), void (*dfunc)(cufftComplex*, float*))
{
  int i = 0;
  double start, end;
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

  normPGC = deviceReduce<float>(device_normVector, M*N);
  i=1;
  while(normPGC > ARMIJOTOLERANCE && i <= it_maximum){
    start = omp_get_wtime();
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

    normPL = deviceReduce<float>(device_normVector, M*N);

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

      normPL = deviceReduce<float>(device_normVector, M*N);
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

    normPGC = deviceReduce<float>(device_normVector, M*N);
    i++;
    end = omp_get_wtime();
    double wall_time = end-start;
    printf("Time: %lf seconds\n", i, wall_time);
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
  double start, end;


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
  if(verbose_flag){
    printf("Starting function value = %f\n", fp);
  }
  (*dfunc)(p,xi);
  //g=-xi
  //xi=h=g
  searchDirection<<<numBlocksNN, threadsPerBlockNN>>>(device_g, xi, device_h, N);//Search direction
  gpuErrchk(cudaDeviceSynchronize());

  ////////////////////////////////////////////////////////////////
  for(int i=1; i <= it_maximum; i++){
    start = omp_get_wtime();
    iter = i;
    if(verbose_flag){
      printf("\n\n********** Iteration %d **********\n\n", i);
    }
    linmin(p, xi, fret, func);

    if (2.0*fabs(*fret-fp) <= ftol*(fabs(*fret)+fabs(fp)+EPS)) {
      printf("Exit due to tolerance\n");
      FREEALL
			return;
		}

    fp=(*func)(p);
    if(verbose_flag){
      printf("Function value = %f\n", fp);
    }
    (*dfunc)(p,xi);
    dgg = gg = 0.0;
    ////gg = g*g
    ////dgg = (xi+g)*xi
    getGandDGG<<<numBlocksNN, threadsPerBlockNN>>>(device_gg_vector, device_dgg_vector, xi, device_g, N);
  	gpuErrchk(cudaDeviceSynchronize());
    ////getSums (Reductions) of gg dgg
    gg = deviceReduce<float>(device_gg_vector, M*N);
    dgg = deviceReduce<float>(device_dgg_vector, M*N);
    if(gg == 0.0){
      printf("Exit due to gg = 0\n");
      FREEALL
      return;
    }
    gam = fmax(0.0f, dgg/gg);
    //printf("Gamma = %f\n", gam);
    //g=-xi
    //xi=h=g+gam*h;
    newXi<<<numBlocksNN, threadsPerBlockNN>>>(device_g, xi, device_h, gam, N);
  	gpuErrchk(cudaDeviceSynchronize());
    end = omp_get_wtime();
    double wall_time = end-start;
    if(verbose_flag){
      printf("Time: %lf seconds\n", i, wall_time);
    }
  }
  printf("Too many iterations in frprmn\n");
  FREEALL
  return;
}

__host__ void LBFGS_recursion(float *d_y, cufftComplex* d_s, float *d_q, int par_M, int M, int N){
  float *alpha, *aux_vector;
  float rho, beta;
  float sy, yy, sy_yy;
  alpha = (float*)malloc(par_M*sizeof(float));

  gpuErrchk(cudaMalloc((void**)&aux_vector, sizeof(float)*M*N));
  gpuErrchk(cudaMemset(aux_vector, 0, sizeof(float)*M*N));



  for(int k=0; k<par_M; k++){
    //Rho_k = 1.0/(y_k's_k);
    getDot_LBFGS_fComplex<<<numBlocksNN, threadsPerBlockNN>>>(aux_vector, d_y, d_s, k, k, M, N);
    gpuErrchk(cudaDeviceSynchronize());
    rho = 1.0/deviceReduce<float>(aux_vector, M*N);

    //alpha_k = Rho_k x (s_k' * q);
    getDot_LBFGS_fComplex<<<numBlocksNN, threadsPerBlockNN>>>(aux_vector, d_s, d_q, k, 0, M, N);
    gpuErrchk(cudaDeviceSynchronize());
    alpha[k] = rho * deviceReduce<float>(aux_vector, M*N);
    //q = q - alpha_k * y_k;
    updateQ<<<numBlocksNN, threadsPerBlockNN>>>(d_q, -alpha[k], d_y, k, N);
    gpuErrchk(cudaDeviceSynchronize());

  }

  //s0'y_0
  getDot_LBFGS_fComplex<<<numBlocksNN, threadsPerBlockNN>>>(aux_vector, d_s, d_y, 0, 0, N);
  gpuErrchk(cudaDeviceSynchronize());
  sy = deviceReduce<float>(aux_vector, M*N);
  //y_0'y_0
  getDot_LBFGS_ff<<<numBlocksNN, threadsPerBlockNN>>>(aux_vector, d_y, d_y, 0, 0, N);
  gpuErrchk(cudaDeviceSynchronize());
  yy = deviceReduce<float>(aux_vector, M*N);
  //(s_0'y_0)/(y_0'y_0)
  sy_yy = sy/yy;
  // r = q x ((s_0'y_0)/(y_0'y_0));
  getR<<<numBlocksNN, threadsPerBlockNN>>>(d_q, sy_yy, N);
  gpuErrchk(cudaDeviceSynchronize());

  for (int k = par_M - 1; k >= 0; k--)
    //Rho_k = 1.0/(y_k's_k);
    getDot_LBFGS_fComplex<<<numBlocksNN, threadsPerBlockNN>>>(aux_vector, d_y, d_s, k, k, N);
    gpuErrchk(cudaDeviceSynchronize());
    //Calculate rho backwards
    rho = 1.0/deviceReduce<float>(aux_vector, M*N);
    //beta = rho * y_k' * r;
    beta = rho * deviceReduce<float>(aux_vector, M*N);
    //r = r + s_k * (alpha_k - beta)
    updateQ<<<numBlocksNN, threadsPerBlockNN>>>(d_q, alpha[k]-beta, d_s, k, N);
  }
  cudaFree(aux_vector);
  free(alpha);
}

__host__ void LBFGS(cufftComplex *p, float ftol, float *fret, float (*func)(cufftComplex*), void (*dfunc)(cufftComplex*, float*), int K)
{
  cufftComplex *d_s;
  float *d_y, *xi, *xi_p, *norm_vector;
  float norm;
  cufftComplex *p_p;

  gpuErrchk(cudaMalloc((void**)&d_y, sizeof(float)*M*N*K));
  gpuErrchk(cudaMemset(d_y, 0, sizeof(float)*M*N*K));

  gpuErrchk(cudaMalloc((void**)&d_s, sizeof(cufftComplex)*M*N*K));
  gpuErrchk(cudaMemset(d_s, 0, sizeof(cufftComplex)*M*N*K));

  gpuErrchk(cudaMalloc((void**)&p_p, sizeof(cufftComplex)*M*N));
  gpuErrchk(cudaMemset(p_p, 0, sizeof(cufftComplex)*M*N));

  gpuErrchk(cudaMalloc((void**)&xi, sizeof(float)*M*N));
  gpuErrchk(cudaMemset(xi, 0, sizeof(float)*M*N));

  gpuErrchk(cudaMalloc((void**)&xi_p, sizeof(float)*M*N));
  gpuErrchk(cudaMemset(xi_p, 0, sizeof(float)*M*N));

  gpuErrchk(cudaMalloc((void**)&norm_vector, sizeof(float)*M*N));
  gpuErrchk(cudaMemset(norm_vector, 0, sizeof(float)*M*N));

  fp = (*func)(p);
  if(verbose_flag){
    printf("Starting function value = %f\n", fp);
  }
  (*dfunc)(p,xi);

  gpuErrchk(cudaMemcpy2D(p_p, sizeof(cufftComplex), p, sizeof(cufftComplex), sizeof(cufftComplex), M*N, cudaMemcpyDeviceToDevice));
  gpuErrchk(cudaMemcpy2D(xi_p, sizeof(float), xi, sizeof(float), sizeof(float), M*N, cudaMemcpyDeviceToDevice));

  searchDirection_LBFGS<<<numBlocksNN, threadsPerBlockNN>>>(xi_p, N);//Search direction
  gpuErrchk(cudaDeviceSynchronize());

  for(int i=1; i <= it_maximum; i++){
    start = omp_get_wtime();
    iter = i;
    if(verbose_flag){
      printf("\n\n********** Iteration %d **********\n\n", i);
    }

    linmin(p_p, xi_p fret, func);

    if (2.0*fabs(*fret-fp) <= ftol*(fabs(*fret)+fabs(fp)+EPS)) {
      printf("Exit due to tolerance\n");
      FREEALL_LBFGS
			return;
		}

    getDot_LBFGS_ff<<<numBlocksNN, threadsPerBlockNN>>>(norm_vector, xi_p, xi_p, 0, 0, M, N);
    gpuErrchk(cudaDeviceSynchronize());
    norm = deviceReduce<float>(norm_vector, M*N);

    if(norm == 0.0){
      printf("Exit due to norm = 0\n");
      FREEALL_LBFGS
      return;
    }

    fp=(*func)(p);
    if(verbose_flag){
      printf("Function value = %f\n", fp);
    }
    (*dfunc)(p,xi);

    calculateSandY<<<numBlocksNN, threadsPerBlockNN>>>(d_s, d_y, p, xi, p_p, xi_p, i-1, M, N);
    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaMemcpy2D(p_p, sizeof(cufftComplex), p, sizeof(cufftComplex), sizeof(cufftComplex), M*N, cudaMemcpyDeviceToDevice));
    gpuErrchk(cudaMemcpy2D(xi_p, sizeof(float), xi, sizeof(float), sizeof(float), M*N, cudaMemcpyDeviceToDevice));

    LBFGS_recursion(d_y, d_s, xi_p, std::min(K,i), M, N);
    searchDirection_LBFGS<<<numBlocksNN, threadsPerBlockNN>>>(xi_p, N);//Search direction
    gpuErrchk(cudaDeviceSynchronize());

    end = omp_get_wtime();
    double wall_time = end-start;
    if(verbose_flag){
      printf("Time: %lf seconds\n", i, wall_time);
    }

  }
  printf("Too many iterations in LBFGS\n");
  FREEALL_LBFGS
  return;

}
