#ifndef IOMS_CUH
#define IOMS_CUH
#include "framework.cuh"
#include "functions.cuh"

class IoMS : public Io
{
public:
freqData IocountVisibilities(char * MS_name, Field *&fields, int gridding);
canvasVariables IoreadCanvas(char *canvas_name, fitsfile *&canvas, float b_noise_aux, int status_canvas, int verbose_flag);
void IoreadMSMCNoise(char *MS_name, Field *fields, freqData data);
void IoreadSubsampledMS(char *MS_name, Field *fields, freqData data, float random_probability);
void IoreadMCNoiseSubsampledMS(char *MS_name, Field *fields, freqData data, float random_probability);
void IoreadMS(char *MS_name, Field *fields, freqData data);
void IowriteMS(char *infile, char *outfile, Field *fields, freqData data, float random_probability, int verbose_flag);
void IocloseCanvas(fitsfile *canvas);
void IoPrintImage(float *I, fitsfile *canvas, char *path, char *name_image, char *units, int iteration, int index, float fg_scale, long M, long N);
void IoPrintImageIteration(float *I, fitsfile *canvas, char *path, char *name_image, char *units, int iteration, int index, float fg_scale, long M, long N);
void IoPrintMEMImageIteration(float *I, char *name_image, char *units, int index);
void doOrderIterations(float *I);
void doOrderEnd(float *I);
void doOrderError(float *I);
};

#endif
