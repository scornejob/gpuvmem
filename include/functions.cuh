#include <math.h>
#include <string.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_constants.h"
#include <float.h>
#include <unistd.h>
#include <getopt.h>
#include <fcntl.h>
#include <omp.h>
#include <string.h>
#include <sys/stat.h>
#include "MSFITSIO.cuh"
#include <cooperative_groups.h>


#define FLOAT_IMG   -32
#define DOUBLE_IMG  -64

#define TSTRING      16
#define TLONG        41
#define TINT         31
#define TFLOAT       42
#define TDOUBLE      82
#define TCOMPLEX     83
#define TDBLCOMPLEX 163

const float PI = CUDART_PI_F;
const double PI_D = CUDART_PI;
const float RPDEG = (PI/180.0);
const double RPDEG_D = (PI_D/180.0);
const float RPARCM = (PI/(180.0*60.0));
const float LIGHTSPEED = 2.99792458E8;
const float RZ = 1.2196698912665045;

typedef struct varsPerGPU{
  float *device_chi2;
  float *device_dchi2;
  cufftHandle plan;
  cufftComplex *device_image;
  cufftComplex *device_V;
}varsPerGPU;

typedef struct variablesPerField{
  float *atten_image;
}VariablesPerField;

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
  int reg_term;
  int gridding;
  float noise;
  float noise_cut;
  float randoms;
  float lambda;
  float minpix;
  float eta;
} Vars;

__host__ void goToError();
__host__ long NearestPowerOf2(long x);

__host__ void readInputDat(char *file);
__host__ void init_beam(int telescope);
__host__ void print_help();
__host__ char *strip(const char *string, const char *chars);
__host__ Vars getOptions(int argc, char **argv);
__host__ float chiCuadrado(cufftComplex *I);
__host__ void dchiCuadrado(cufftComplex *I, float *dxi2);
__host__ void do_gridding(Field *fields, MSData *data, float deltau, float deltav, int M, int N, int *total_visibilities);
__host__ float calculateNoise(Field *fields, MSData data, int *total_visibilities, int blockSizeV);
__host__ void clipping(cufftComplex *I, int iterations);
template <class T>
__host__ T deviceReduce(T *in, long N);



__global__ void deviceReduceKernel(float *g_idata, float *g_odata, unsigned int n);
__global__ void clipWNoise(cufftComplex *fg_image, float *noise, cufftComplex *I, long N, float noise_cut, float MINPIX);
__global__ void getGandDGG(float *gg, float *dgg, float *xi, float *g, long N);
__global__ void newP(cufftComplex *p, float *xi, float xmin, float MINPIX, float eta, long N);
__global__ void newPNoPositivity(cufftComplex *p, float *xi, float xmin, long N);
__global__ void clip(cufftComplex *I, long N, float MINPIX);
__global__ void hermitianSymmetry(float *Ux, float *Vx, cufftComplex *Vo, float freq, int numVisibilities);
__device__ float attenuation(float antenna_diameter, float pb_factor, float pb_cutoff, float freq, float xobs, float yobs, float DELTAX, float DELTAY);
__global__ void total_attenuation(float *total_atten, float beam_fwhm, float beam_freq, float beam_cutoff, float freq, float xobs, float yobs, float DELTAX, float DELTAY, long N);
__global__ void mean_attenuation(float *total_atten, int channels, long N);
__global__ void weight_image(float *weight_image, float *total_atten, float noise_jypix, long N);
__global__ void noise_image(float *noise_image, float *weight_image, float noise_jypix, long N);
__global__ void apply_beam(float antenna_diameter, float pb_factor, float pb_cutoff, cufftComplex *image, cufftComplex *fg_image, long N, float xobs, float yobs, float fg_scale, float freq, float DELTAX, float DELTAY);
__global__ void phase_rotate(cufftComplex *data, long M, long N, float xphs, float yphs);
__global__ void vis_mod(cufftComplex *Vm, cufftComplex *V, float *Ux, float *Vx, float *weight, float deltau, float deltav, long numVisibilities, long N);
__global__ void residual(cufftComplex *Vr, cufftComplex *Vm, cufftComplex *Vo, long numVisibilities);
__global__ void makePositive(cufftComplex *I, long N);
__global__ void evaluateXt(cufftComplex *xt, cufftComplex *pcom, float *xicom, float x, float MINPIX, float eta, long N);
__global__ void evaluateXtNoPositivity(cufftComplex *xt, cufftComplex *pcom, float *xicom, float x, long N);
__global__ void chi2Vector(float *chi2, cufftComplex *Vr, float *w, int numVisibilities);
__global__ void SVector(float *S, float *noise, cufftComplex *I, long N, float noise_cut, float MINPIX, float eta);
__global__ void QPVector(float *Q, float *noise, cufftComplex *I, long N, float noise_cut, float MINPIX);
__global__ void TVVector(float *TV, float *noise, cufftComplex *I, long N, float noise_cut, float MINPIX);
__global__ void searchDirection(float *g, float *xi, float *h, long N);
__global__ void newXi(float *g, float *xi, float *h, float gam, long N);
__global__ void clip(cufftComplex *I, float *grad, float RMS, long N);
__global__ void DS(float *dH, cufftComplex *I, float *noise, float noise_cut, float lambda, float MINPIX, float eta, long N);
__global__ void DQ(float *dQ, cufftComplex *I, float *noise, float noise_cut, float lambda, float MINPIX, long N);
__global__ void DTV(float *dTV, cufftComplex *I, float *noise, float noise_cut, float lambda, float MINPIX, long N);
__global__ void DChi2(float *noise, float *dChi2, cufftComplex *Vr, float *U, float *V, float *w, long N, long numVisibilities, float fg_scale, float noise_cut, float xobs, float yobs, float DELTAX, float DELTAY, float antenna_diameter, float pb_factor, float pb_cutoff, float freq);
__global__ void DPhi(float *dphi, float *dchi2, float *dH, float lambda, long N);
__global__ void projection(float *px, float *x, float MINPIX, long N);
__global__ void substraction(float *x, cufftComplex *xc, float *gc, float lambda, long N);
__global__ void normVectorCalculation(float *normVector, float *gc, long N);
__global__ void copyImage(cufftComplex *p, float *device_xt, long N);
__global__ void searchDirection_LBFGS(float *xi, long N);
__global__ void getDot_LBFGS_fComplex (float *aux_vector, cufftComplex *vec_1, float *vec_2, int k, int h, int M, int N);
__global__ void getDot_LBFGS_ff (float *aux_vector, float *vec_1, float *vec_2, int k, int h, int M, int N);
__global__ void updateQ (float *d_q, float alpha, float *d_y, int k, int M, int N);
__global__ void updateQComplex (float *d_q, float alpha, cufftComplex *d_y, int k, int M, int N);
__global__ void getR (float *d_q, float scalar, int N);
__global__ void calculateSandY (cufftComplex *d_s, float *d_y, cufftComplex *p, float *xi, cufftComplex *p_p, float *xi_p, int iter, int M, int N);
