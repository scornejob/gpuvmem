#include "imageProcessor.cuh"

extern int image_count;
extern long N, M;

ImageProcessor::ImageProcessor()
{
}

void ImageProcessor::clip(float *I)
{
  if(image_count == 1)
  {
    linkClip(I);
  }

  if(image_count == 2)
  {
    linkClip(I);
  }
};
void ImageProcessor::calculateInu(cufftComplex *image, float *I, float freq)
{
  if(image_count == 2)
  {
    linkCalculateInu2I(image, I, freq);
  }
};

void ImageProcessor::apply_beam(cufftComplex *image, float xobs, float yobs, float freq)
{
  if(image_count == 1)
    linkApplyBeam1I(image, fg_image, xobs, yobs, freq);
  if(image_count == 2)
    linkApplyBeam2I(image, xobs, yobs, freq);
};

void ImageProcessor::chainRule(float *I, float freq)
{
  if(image_count == 2)
  {
    linkChain2I(chain, freq, I);
  }

};

void ImageProcessor::clip(cufftComplex *I)
{
  if(image_count == 1)
  {

  }
};

void ImageProcessor::clipWNoise(float *I)
{
  if(image_count == 1)
    linkClipWNoise1I(fg_image, I);
  if(image_count == 2)
    linkClipWNoise2I(I);
};

void ImageProcessor::configure(int I)
{
  this->image_count = I;
  if(image_count == 1)
  {
    gpuErrchk(cudaMalloc((void**)&fg_image, sizeof(cufftComplex)*M*N));
    gpuErrchk(cudaMemset(fg_image, 0, sizeof(cufftComplex)*M*N));
  }

  if(image_count > 1)
  {
    gpuErrchk(cudaMalloc((void**)&chain, sizeof(float)*M*N*image_count));
    gpuErrchk(cudaMemset(chain, 0, sizeof(float)*M*N*image_count));
  }
};
