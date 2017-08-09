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

#include "linmin.cuh"
#define TOL 1.0e-7

float3 *device_pcom;
float3 *device_xicom;
float (*nrfunc)(float3*);
extern long M;
extern long N;
extern float MINPIX, beta_start, T_min, tau_min;
extern int nopositivity;

extern dim3 threadsPerBlockNN;
extern dim3 numBlocksNN;
extern int verbose_flag;


__host__ void linmin(float3 *p, float3 *xi, float *fret, float (*func)(float3*))//p and xi are in GPU
{
  float xx, xmin, fx, fb, fa, bx ,ax;

  gpuErrchk(cudaMalloc((void**)&device_pcom, sizeof(float3)*M*N));
  gpuErrchk(cudaMemset(device_pcom, 0, sizeof(float3)*M*N));

  gpuErrchk((cudaMalloc((void**)&device_xicom, sizeof(float3)*M*N)));
  gpuErrchk(cudaMemset(device_xicom, 0, sizeof(float3)*M*N));
  nrfunc = func;
  //device_pcom = p;
  //device_xicom = xi;
  gpuErrchk(cudaMemcpy2D(device_pcom, sizeof(float3), p, sizeof(float3), sizeof(float3), M*N, cudaMemcpyDeviceToDevice));
  gpuErrchk(cudaMemcpy2D(device_xicom, sizeof(float3), xi, sizeof(float3), sizeof(float3), M*N, cudaMemcpyDeviceToDevice));

  ax = 0.0;
	xx = 1.0;

  mnbrak(&ax, &xx, &bx, &fa, &fx, &fb, f1dim);


  *fret = brent(ax, xx, bx, TOL, &xmin, f1dim);
  if(verbose_flag){
    printf("xmin = %f\n\n", xmin);
  }

  //GPU MUL AND ADD
  //xi     = xi*xmin;
  //p      = p + xi;
  if(nopositivity == 0){
    newP<<<numBlocksNN, threadsPerBlockNN>>>(p, xi, xmin, N, tau_min, T_min, beta_start);
    gpuErrchk(cudaDeviceSynchronize());
  }else{
    newPNoPositivity<<<numBlocksNN, threadsPerBlockNN>>>(p, xi, xmin, N);
    gpuErrchk(cudaDeviceSynchronize());
  }

  cudaFree(device_xicom);
  cudaFree(device_pcom);
}
#undef TOL
