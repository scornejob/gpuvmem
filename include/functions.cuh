#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>
#include <string.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_constants.h"
#include "rngs.cuh"
#include <cufft.h>
#include "fitsio.h"
#include <float.h>
#include <unistd.h>
#include <getopt.h>
#include <fcntl.h>
#include <omp.h>
#include <string.h>
#include <sys/stat.h>
#include <tables/Tables/Table.h>
#include <tables/Tables/TableRow.h>
#include <tables/Tables/TableIter.h>
#include <tables/Tables/ScalarColumn.h>
#include <tables/Tables/ArrayColumn.h>
#include <casa/Arrays/Vector.h>
#include <casa/Arrays/Slicer.h>
#include <casa/Arrays/ArrayMath.h>
#include <tables/Tables/TableParse.h>
#include <ms/MeasurementSets.h>
#include <tables/Tables/ColumnDesc.h>
#include <tables/Tables/ScaColDesc.h>
#include <tables/Tables/ArrColDesc.h>
#include <ms/MeasurementSets/MSMainColumns.h>
#include <tables/Tables/TableDesc.h>
#include <ms/MeasurementSets/MSAntennaColumns.h>

#define FLOAT_IMG   -32
#define DOUBLE_IMG  -64

#define TSTRING      16
#define TLONG        41
#define TINT         31
#define TFLOAT       42
#define TDOUBLE      82
#define TCOMPLEX     83
#define TDBLCOMPLEX 163
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

const float PI = CUDART_PI_F;
const double PI_D = CUDART_PI;
const float RPDEG = (PI/180.0);
const double RPDEG_D = (PI_D/180.0);
const float RPARCM = (PI/(180.0*60.0));
const float LIGHTSPEED = 2.99792458E8;

typedef struct observedVisibilities{
  float *u;
  float *v;
  float *weight;
  cufftComplex *Vo;
  cufftComplex *Vm;
  cufftComplex *Vr;
  float freq;
  long numVisibilities;

  int *stokes;
  int threadsPerBlockUV;
  int numBlocksUV;
}Vis;

typedef struct variablesPerFreq{
  cufftComplex *atten;
  float *chi2;
  float *dchi2;
  float alpha;
  float *alpha_num;
  float *alpha_den;
  cufftHandle plan;
  cufftComplex *device_image;
  cufftComplex *device_V;
}VPF;

typedef struct freqData{
  int n_internal_frequencies;
  int total_frequencies;
  int *channels;
}freqData;

typedef struct field{
  int valid_frequencies;
  double obsra, obsdec;
  float global_xobs, global_yobs;
  long *numVisibilitiesPerFreq;
  cufftComplex *atten_image;
  VPF *device_vars;
  Vis *visibilities;
  Vis *device_visibilities;
}Field;

typedef struct variables {
	char *input;
  char *output;
  char *inputdat;
  char *modin;
  char *ofile;
  char *path;
  char *output_image;
  char *multigpu;
  int select;
  int blockSizeX;
  int blockSizeY;
  int blockSizeV;
  int it_max;
  float noise;
  float noise_cut;
  float randoms;
  float lambda;
  float minpix;
} Vars;

__host__ void goToError();
__host__ freqData getFreqs(char * file);
__host__ long NearestPowerOf2(long N);
__host__ void readInputDat(char *file);
__host__ void init_beam(int telescope);
__host__ void residualsToHost(Field *fields, freqData data);
__host__ void readMS(char *file, char *file2, Field *fields);
__host__ void MScopy(char const *in_dir, char const *in_dir_dest);
__host__ void writeMS(char *infile, char *outfile, Field *fields);
__host__ void print_help();
__host__ char *strip(const char *string, const char *chars);
__host__ Vars getOptions(int argc, char **argv);
__host__ void Print2DFloatArray(int rows, int cols, float *array);
__host__ void Print2DIntArray(int rows, int cols, int *array);
__host__ void Print2DComplex(int rows, int cols, cufftComplex *data, bool cufft_symmetry);
__host__ void toFitsFloat(cufftComplex *I, int iteration, long M, long N, int option);
__host__ float chiCuadrado(cufftComplex *I);
__host__ void dchiCuadrado(cufftComplex *I, float *dxi2);
__host__ void clipping(cufftComplex *I, int iterations);
__host__ float deviceReduce(float *in, long N);



__global__ void deviceReduceKernel(float *g_idata, float *g_odata, unsigned int n);
__global__ void clipWNoise(cufftComplex *fg_image, cufftComplex *noise, cufftComplex *I, long N, float noise_cut, float MINPIX);
__global__ void getGandDGG(float *gg, float *dgg, float *xi, float *g, long N);
__global__ void newP(cufftComplex *p, float *xi, float xmin, float MINPIX, long N);
__global__ void newPNoPositivity(cufftComplex *p, float *xi, float xmin, long N);
__global__ void clip(cufftComplex *I, long N, float MINPIX);
__global__ void hermitianSymmetry(float *Ux, float *Vx, cufftComplex *Vo, float freq, int numVisibilities);
__global__ void attenuation(float beam_fwhm, float beam_freq, float beam_cutoff, cufftComplex *attenMatrix, float freq, long N, float xobs, float yobs, float DELTAX, float DELTAY);
__global__ void total_attenuation(cufftComplex *total_atten, cufftComplex *attenperFreq, long N);
__global__ void mean_attenuation(cufftComplex *total_atten, int channels, long N);
__global__ void noise_image(cufftComplex *noise_image, cufftComplex *weight_image, float difmap_noise, long N);
__global__ void weight_image(cufftComplex *weight_image, cufftComplex *total_atten, float difmap_noise, long N);
__global__ void apply_beam(float beam_fwhm, float beam_freq, float beam_cutoff, cufftComplex *image, cufftComplex *fg_image, long N, float xobs, float yobs, float fg_scale, float freq, float DELTAX, float DELTAY);
__global__ void phase_rotate(cufftComplex *data, long M, long N, float xphs, float yphs);
__global__ void vis_mod(cufftComplex *Vm, cufftComplex *Vo, cufftComplex *V, float *Ux, float *Vx, float deltau, float deltav, long numVisibilities, long N);
__global__ void alphaVectors(float *alpha_num, float *alpha_den, float *w, cufftComplex *Vm, cufftComplex *Vo, long numVisibilities);
__global__ void residual(cufftComplex *Vr, cufftComplex *Vm, cufftComplex *Vo, long numVisibilities);
__global__ void residual_XCORR(cufftComplex *Vr, cufftComplex *Vm, cufftComplex *Vo, float alpha, long numVisibilities);
__global__ void makePositive(cufftComplex *I, long N);
__global__ void evaluateXt(cufftComplex *xt, cufftComplex *pcom, float *xicom, float x, float MINPIX, long N);
__global__ void evaluateXtNoPositivity(cufftComplex *xt, cufftComplex *pcom, float *xicom, float x, long N);
__global__ void chi2Vector(float *chi2, cufftComplex *Vr, float *w, int numVisibilities);
__global__ void SVector(float *H, cufftComplex *noise, cufftComplex *I, long N, float noise_cut, float MINPIX);
__global__ void searchDirection(float *g, float *xi, float *h, long N);
__global__ void newXi(float *g, float *xi, float *h, float gam, long N);
__global__ void clip(cufftComplex *I, float *grad, float RMS, long N);
__global__ void restartDPhi(float *dphi, float *dChi2, float *dH, long N);
__global__ void DS(float *dH, cufftComplex *I, cufftComplex *noise, float noise_cut, float lambda, float MINPIX, long N);
__global__ void DChi2(cufftComplex *noise, cufftComplex *atten, float *dChi2, cufftComplex *Vr, float *U, float *V, float *w, long N, long numVisibilities, float fg_scale, float noise_cut, float xobs, float yobs, float DELTAX, float DELTAY);
__global__ void DChi2_XCORR(cufftComplex *noise, cufftComplex *atten, float *dChi2, cufftComplex *Vr, float *U, float *V, float *w, long N, long numVisibilities, float fg_scale, float noise_cut, float xobs, float yobs, float alpha, float DELTAX, float DELTAY);
__global__ void DPhi(float *dphi, float *dchi2, float *dH, float lambda, long N);
__global__ void projection(float *px, float *x, float MINPIX, long N);
__global__ void substraction(float *x, cufftComplex *xc, float *gc, float lambda, long N);
__global__ void normVectorCalculation(float *normVector, float *gc, long N);
__global__ void copyImage(cufftComplex *p, float *device_xt, long N);
