#ifndef FUNCTIONS_CUH
#define FUNCTIONS_CUH

#include "framework.cuh"
#include "copyrightwarranty.cuh"

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
const float RPARCSEC = (PI/(180.0*3600.0));
const float RPARCM = (PI/(180.0*60.0));
const float LIGHTSPEED = 2.99792458E8;
const float RZ = 1.2196698912665045;

__host__ void goToError();
__host__ long NearestPowerOf2(long x);

__host__ void readInputDat(char *file);
__host__ void init_beam(int telescope);
__host__ void print_help();
__host__ char *strip(const char *string, const char *chars);
__host__ Vars getOptions(int argc, char **argv);
__host__ float chiCuadrado(float *I);
__host__ void dchiCuadrado(float *I, float *dxi2);
__host__ void do_gridding(Field *fields, freqData *data, float deltau, float deltav, int M, int N, float robust);
__host__ void degridding(Field *fields, freqData data, float deltau, float deltav, int num_gpus, int firstgpu, int blockSizeV, long M, long N);
__host__ float calculateNoise(Field *fields, freqData data, int *total_visibilities, int blockSizeV, int gridding);
__host__ void clipping(cufftComplex *I, int iterations);
template <class T>
__host__ T deviceReduce(T *in, long N);
__host__ float chi2(float *I, VirtualImageProcessor *ip);
__host__ void linkRestartDGi(float *dgi);
__host__ void linkAddToDPhi(float *dphi, float *dgi, int index);
__host__ void dchi2(float *I, float *dxi2, float *result_dchi2, VirtualImageProcessor *ip);
__host__ float laplacian(float *I, float * ds, float penalization_factor, int mod, int order, int imageIndex);
__host__ void DLaplacian(float *I, float *dgi, float penalization_factor, float mod, float order, float index);
__host__ void defaultNewP(float*p, float*xi, float xmin, int image);
__host__ void particularNewP(float*p, float*xi, float xmin, int image);
__host__ void defaultEvaluateXt(float*xt, float*pcom, float*xicom, float x, int image);
__host__ void particularEvaluateXt(float*xt, float*pcom, float*xicom, float x, int image);
__host__ void linkApplyBeam2I(cufftComplex *image, float xobs, float yobs, float freq);
__host__ void linkClipWNoise2I(float *I);
__host__ void linkCalculateInu2I(cufftComplex *image, float *I, float freq);
__host__ void linkChain2I(float *chain, float freq, float *I);
__host__ void linkClip(float *I);
__host__ void DEntropy(float *I, float *dgi, float penalization_factor, int mod, int order, int index);
__host__ float SEntropy(float *I, float * ds, float penalization_factor, int mod, int order, int index);
__host__ void DTVariation(float *I, float *dgi, float penalization_factor, int mod, int order, int index);
__host__ float totalvariation(float *I, float * ds, float penalization_factor, int mod, int order, int index);
__host__ void DQuadraticP(float *I, float *dgi, float penalization_factor, int mod, int order, int index);
__host__ float quadraticP(float *I, float * ds, float penalization_factor, int mod, int order, int index);
__host__ float squaredTotalVariation(float *I, float * ds, float penalization_factor, int mod, int order, int index);
__host__ void DSTVariation(float *I, float *dgi, float penalization_factor, int mod, int order, int index);
__host__ void calculateErrors(Image *image);


__global__ void deviceReduceKernel(float *g_idata, float *g_odata, unsigned int n);
__global__ void clipWNoise(cufftComplex *fg_image, float *noise, float *I, long N, float noise_cut, float MINPIX);
__global__ void getGandDGG(float *gg, float *dgg, float *xi, float *g, long N);
__global__ void newP(float *p, float *xi, float xmin, float MINPIX, float eta, long N);
__global__ void newPNoPositivity(cufftComplex *p, float *xi, float xmin, long N);
__global__ void clip(cufftComplex *I, long N, float MINPIX);
__global__ void hermitianSymmetry(float *Ux, float *Vx, cufftComplex *Vo, float freq, int numVisibilities);
__device__ float attenuation(float beam_fwhm, float beam_freq, float beam_cutoff, float freq, float xobs, float yobs, float DELTAX, float DELTAY);
__global__ void total_attenuation(float *total_atten, float antenna_diameter, float pb_factor, float pb_cutoff, float freq, float xobs, float yobs, float DELTAX, float DELTAY, long N);
__global__ void mean_attenuation(float *total_atten, int channels, long N);
__global__ void weight_image(float *weight_image, float *total_atten, float noise_jypix, long N);
__global__ void noise_image(float *noise_image, float *weight_image, float noise_jypix, long N);
__global__ void phase_rotate(cufftComplex *data, long M, long N, float xphs, float yphs);
__global__ void vis_mod(cufftComplex *Vm, cufftComplex *V, float *Ux, float *Vx, float *weight, float deltau, float deltav, long numVisibilities, long N);
__global__ void vis_mod2(cufftComplex *Vm, cufftComplex *V, float *Ux, float *Vx, float *weight, float deltau, float deltav, long numVisibilities, long N);
__global__ void vis_mod3(cufftComplex *Vm, cufftComplex *V, float *Ux, float *Vx, float *weight, float deltau, float deltav, long numVisibilities, long N);
__global__ void residual(cufftComplex *Vr, cufftComplex *Vm, cufftComplex *Vo, long numVisibilities);
__global__ void makePositive(cufftComplex *I, long N);
__global__ void evaluateXt(float *xt, float *pcom, float *xicom, float x, float MINPIX, float eta, long N);
__global__ void evaluateXtNoPositivity(cufftComplex *xt, cufftComplex *pcom, float *xicom, float x, long N);
__global__ void chi2Vector(float *chi2, cufftComplex *Vr, float *w, int numVisibilities);
__global__ void SVector(float *S, float *noise, cufftComplex *I, long N, float noise_cut, float MINPIX, float eta);
__global__ void QPVector(float *Q, float *noise, float *I, long N, float noise_cut, float MINPIX);
__global__ void TVVector(float *TV, float *noise, float *I, long N, float noise_cut, float MINPIX);
__global__ void searchDirection(float *g, float *xi, float *h, long N);
__global__ void newXi(float *g, float *xi, float *h, float gam, long N);
__global__ void clip(cufftComplex *I, float *grad, float RMS, long N);
__global__ void restartDPhi(float *dphi, float *dChi2, float *dH, long N);
__global__ void DS(float *dH, cufftComplex *I, float *noise, float noise_cut, float lambda, float MINPIX, float eta, long N);
__global__ void DQ(float *dQ, cufftComplex *I, float *noise, float noise_cut, float lambda, float MINPIX, long N);
__global__ void DTV(float *dTV, cufftComplex *I, float *noise, float noise_cut, float lambda, float MINPIX, long N);
__global__ void DChi2(float *noise, float *dChi2, cufftComplex *Vr, float *U, float *V, float *w, long N, long numVisibilities, float fg_scale, float noise_cut, float xobs, float yobs, float DELTAX, float DELTAY, float antenna_diameter, float pb_factor, float pb_cutoff, float freq);
__global__ void DPhi(float *dphi, float *dchi2, float *dH, float lambda, long N);
__global__ void projection(float *px, float *x, float MINPIX, long N);
__global__ void substraction(float *x, cufftComplex *xc, float *gc, float lambda, long N);
__global__ void normVectorCalculation(float *normVector, float *gc, long N);
__global__ void copyImage(cufftComplex *p, float *device_xt, long N);
__global__ void searchDirection(float* g, float* xi, float* h, long N, long M, int image);
__global__ void getGGandDGG(float *gg, float *dgg, float* xi, float* g, long N, long M, int image);
__global__ void newXi(float* g, float* xi, float* h, float gam, long N, long M, int image);
__global__ void evaluateXt(float*xt, float*pcom, float*xicom, float x, long N, long M, float MINPIX, float eta, int image);
__global__ void evaluateXtDefault(float*xt, float*pcom, float*xicom, float x, long N, long M, float MINPIX, float eta, int image);
__global__ void newP(float*p, float*xi, float xmin, long N, long M, float MINPIX, float eta, int image);
__global__ void newPNoPositivity(float *p, float *xi, float xmin, long N, long M, int image);
__global__ void evaluateXtNoPositivity(float *xt, float *pcom, float *xicom, float x, long N, long M, int image);
__global__ void chainRule2I(float *chain, float *noise, float *I, float nu, float nu_0, float noise_cut, float fg_scale, long N, long M);
__global__ void DChi2_2I(float *noise, float *chain, float *dchi2, float *dchi2_total, float threshold, int image, long N, long M);
__global__ void calculateSandY (float *d_y, float *d_s, float *p, float *xi, float *p_old, float *xi_old, int iter, int M, int N, int image);
__global__ void getR (float *d_r, float *d_q, float scalar, int M, int N, int image);
__global__ void updateQ (float *d_q, float alpha, float *d_y, int k, int M, int N, int image);
__global__ void getDot_LBFGS_ff(float *aux_vector, float *vec_1, float *vec_2, int k, int h, int M, int N, int image);
__global__ void searchDirection_LBFGS(float *xi, long N, long M, int image);
__global__ void fftshift_2D(cufftComplex *data, int N1, int N2);


#endif
