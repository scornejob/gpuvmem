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

#define ALPHA 1.e-4


#define FREEALL cudaFree(device_gg_vector);cudaFree(device_dgg_vector);cudaFree(xi);cudaFree(device_h);cudaFree(device_g);


__host__ void frprmn(float3 *p, float ftol, float *fret, float (*func)(float3*), void (*dfunc)(float3*, float3*))
{
  float gg, dgg, gam, fp;
  float3 *device_g, *device_h, *xi;
  double start, end;


  //////////////////////MEMORY GPU//////////////////////////
  gpuErrchk(cudaMalloc((void**)&device_g, sizeof(float3)*M*N));
  gpuErrchk(cudaMemset(device_g, 0, sizeof(float3)*M*N));
  gpuErrchk(cudaMalloc((void**)&device_h, sizeof(float3)*M*N));
  gpuErrchk(cudaMemset(device_h, 0, sizeof(float3)*M*N));
  gpuErrchk(cudaMalloc((void**)&xi, sizeof(float3)*M*N));
  gpuErrchk(cudaMemset(xi, 0, sizeof(float3)*M*N));

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
    getGGandDGG<<<numBlocksNN, threadsPerBlockNN>>>(device_gg_vector, device_dgg_vector, xi, device_g, N);
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
