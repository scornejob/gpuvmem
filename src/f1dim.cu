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

#include "f1dim.cuh"
extern float2 *device_pcom;
extern float2 *device_xicom;
extern float (*nrfunc)(float2*);
extern long M;
extern long N;
extern float MINPIX, eta, fg_scale;
extern dim3 threadsPerBlockNN;
extern dim3 numBlocksNN;
extern int nopositivity;

__host__ float f1dim(float x)
{
    float2 *device_xt;
    float f;

    gpuErrchk(cudaMalloc((void**)&device_xt, sizeof(float2)*M*N));
    gpuErrchk(cudaMemset(device_xt, 0, sizeof(float2)*M*N));

    //xt = pcom+x*xicom;
    if(nopositivity == 0){
      evaluateXt<<<numBlocksNN, threadsPerBlockNN>>>(device_xt, device_pcom, device_xicom, x, N, MINPIX, fg_scale, eta);
      gpuErrchk(cudaDeviceSynchronize());
    }else{
      evaluateXtNoPositivity<<<numBlocksNN, threadsPerBlockNN>>>(device_xt, device_pcom, device_xicom, x, N);
      gpuErrchk(cudaDeviceSynchronize());
    }

    f = (*nrfunc)(device_xt);
    cudaFree(device_xt);
    return f;
}
