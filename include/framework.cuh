#ifndef FRAMEWORK_CUH
#define FRAMEWORK_CUH

#include <vector>
#include <iostream>
#include <map>
#include <algorithm>
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
#include <sys/stat.h>
#include "MSFITSIO.cuh"
#include <cooperative_groups.h>

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
  char *initial_values;
  char *penalization_factors;
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
  float eta;
  float nu_0;
  float threshold;
} Vars;

typedef struct functionMap
{
  void (*newP)(float*, float*, float, int);
  void (*evaluateXt)(float*, float*, float*, float, int);
} imageMap;

class VirtualImageProcessor
{
public:
  virtual void clip(float *I) = 0;
  virtual void clip(cufftComplex *I) = 0;
  virtual void clipWNoise(float *I) = 0;
  virtual void apply_beam(cufftComplex *image, float xobs, float yobs, float freq) = 0;
  virtual void calculateInu(cufftComplex *image, float *I, float freq) = 0;
  virtual void chainRule(float *I, float freq) = 0;
  virtual void configure(int I) = 0;
public:
  float *chain;
  cufftComplex *fg_image;
  int image_count;
};


template<class T>
class Singleton
{
public:
  static T& Instance()
  {
    static T instance;
    return instance;
  }
private:
  Singleton(){};
  Singleton(T const&)         = delete;
  void operator=(T const&) = delete;
};

class Fi
{
public:
  virtual float calcFi(float *p) = 0;
  virtual void calcGi(float *p, float *xi) = 0;
  virtual void restartDGi() = 0;
  virtual void addToDphi(float *device_dphi) = 0;
  virtual void configure(int penalizatorIndex, int imageIndex, int imageToAdd) = 0;
  void setPenalizationFactor(float p){this->penalization_factor = p;};
  void setInu(cufftComplex *Inu){this->Inu = Inu;}
  cufftComplex *getInu(){return this->Inu;}
  void setS(float *S){cudaFree(device_S);this->device_S = S;};
  void setDS(float *DS){cudaFree(device_DS);this->device_DS = DS;};
  virtual float calculateSecondDerivate() = 0;
protected:
  float *device_S;
  float *device_DS;
  float penalization_factor = 1;
  int imageIndex;
  int mod;
  int order;
  cufftComplex * Inu = NULL;
  int imageToAdd;
};


class Image
{
public:
  Image(float *image, int image_count){this->image = image; this->image_count = image_count;};
  int image_count;
  float *image;
  imageMap *functionMapping;
  float *error_image;
};

class Visibilities
{
public:
  freqData *data;
  VariablesPerField *vars_per_field;
  Field *fields;
  int *total_visibilities;
};

class Error
{
public:
  virtual void calculateErrorImage(Image *I, Visibilities *v) = 0;
};

class ObjectiveFunction
{
public:
  ObjectiveFunction(){};
  void addFi(Fi *fi){fis.push_back(fi);};
  //virtual void print() = 0;
  float calcFunction(float *p)
  {
    float value = 0.0;

    for(vector<Fi*>::iterator it = fis.begin(); it != fis.end(); it++)
    {
      value += (*it)->calcFi(p);
    }

    return value;
  };

  void calcGradient(float *p, float *xi)
  {
    restartDPhi();
    for(vector<Fi*>::iterator it = fis.begin(); it != fis.end(); it++)
    {
      (*it)->calcGi(p, xi);
      (*it)->addToDphi(dphi);
    }
    phiStatus = 1;
    copyDphiToXi(xi);
  };

  void restartDPhi()
  {
    for(vector<Fi*>::iterator it = fis.begin(); it != fis.end(); it++)
    {
      (*it)->restartDGi();
    }
    gpuErrchk(cudaMemset(dphi, 0, sizeof(float)*M*N*image_count));
  }

  void copyDphiToXi(float *xi)
  {
    gpuErrchk(cudaMemcpy(xi, dphi, sizeof(float)*M*N*image_count, cudaMemcpyDeviceToDevice));
  }
  void setN(long N){this->N = N;}
  void setM(long M){this->M = M;}
  void setImageCount(int I){this->image_count = I;}
  void configure(long N, long M, int I)
  {
      setN(N);
      setM(M);
      setImageCount(I);
      gpuErrchk(cudaMalloc((void**)&dphi, sizeof(float)*M*N*I));
      gpuErrchk(cudaMemset(dphi, 0, sizeof(float)*M*N*I));
  }
private:
  vector<Fi*> fis;
  float *dphi;
  int phiStatus = 1;
  int flag = 0;
  long N = 0;
  long M = 0;
  int image_count = 1;
};

class Optimizator
{
public:
  __host__ virtual void allocateMemoryGpu() = 0;
  __host__ virtual void deallocateMemoryGpu() = 0;
  __host__ virtual void minimizate() = 0;
  //__host__ virtual void configure() = 0;
  __host__ void setImage(Image *image){this->image = image;};
  __host__ void setObjectiveFunction(ObjectiveFunction *of){ this->of = of;};
  void setFlag(int flag){this->flag = flag;};
protected:
  ObjectiveFunction *of;
  Image *image;
  int flag;
};



class Filter
{
public:
  virtual void applyCriteria(Visibilities *v) = 0;
  virtual void configure(void *params) = 0;
};

class Io
{
public:
  virtual freqData IocountVisibilities(char * MS_name, Field *&fields) = 0;
  virtual canvasVariables IoreadCanvas(char *canvas_name, fitsfile *&canvas, float b_noise_aux, int status_canvas, int verbose_flag) = 0;
  virtual void IoreadMSMCNoise(char *MS_name, Field *fields, freqData data) = 0;
  virtual void IoreadSubsampledMS(char *MS_name, Field *fields, freqData data, float random_probability) = 0;
  virtual void IoreadMCNoiseSubsampledMS(char *MS_name, Field *fields, freqData data, float random_probability) = 0;
  virtual void IoreadMS(char *MS_name, Field *fields, freqData data) = 0;
  virtual void IowriteMS(char *infile, char *outfile, Field *fields, freqData data, float random_probability, int verbose_flag) = 0;
  virtual void IocloseCanvas(fitsfile *canvas) = 0;
};

//Implementation of Factory
class Synthesizer
{
public:
  __host__ virtual void run() = 0;
  __host__ virtual void setOutPut(char * FileName) = 0;
  __host__ virtual void setDevice() = 0;
  __host__ virtual void unSetDevice() = 0;
  __host__ virtual void configure(int argc, char **argv) = 0;
  __host__ virtual void applyFilter(Filter *filter) = 0;
  __host__ void setOptimizator(Optimizator *min){this->optimizator = min;};
  __host__ void setVisibilities(Visibilities * v){this->visibilities = v;};
  __host__ void setIoHandler(Io *handler){this->iohandler = handler;};
  __host__ void setError(Error *e){this->error = e;};
  cufftComplex *device_I;
  float ftol;
  Image *image;
protected:
  Optimizator *optimizator;
  Io *iohandler = NULL;
  Visibilities *visibilities;
  Error *error = NULL;
};



class SynthesizerFactory
{
public:
  typedef Synthesizer* (*CreateSynthesizerCallback)();
private:
  typedef map<int, CreateSynthesizerCallback> CallbackMap;
public:
  // Returns true if registration was succesfull
  bool RegisterSynthesizer(int SynthesizerId, CreateSynthesizerCallback CreateFn)
  {
    return callbacks_.insert(CallbackMap::value_type(SynthesizerId, CreateFn)).second;
  };

  bool UnregisterSynthesizer(int SynthesizerId)
  {
    return callbacks_.erase(SynthesizerId) == 1;
  };

  Synthesizer* CreateSynthesizer(int SynthesizerId)
  {
    CallbackMap::const_iterator i = callbacks_.find(SynthesizerId);
     if (i == callbacks_.end())
     {
     // not found
     throw std::runtime_error("Unknown Synthesizer ID");
     }
     // Invoke the creation function
     return (i->second)();
  };

private:
  CallbackMap callbacks_;
};

class FiFactory
{
public:
  typedef Fi* (*CreateFiCallback)();
private:
  typedef map<int, CreateFiCallback> CallbackMap;
public:
  // Returns true if registration was succesfull
  bool RegisterFi(int FiId, CreateFiCallback CreateFn)
  {
    return callbacks_.insert(CallbackMap::value_type(FiId, CreateFn)).second;
  };

  bool UnregisterFi(int FiId)
  {
    return callbacks_.erase(FiId) == 1;
  };

  Fi* CreateFi(int FiId)
  {
    CallbackMap::const_iterator i = callbacks_.find(FiId);
     if (i == callbacks_.end())
     {
     // not found
     throw std::runtime_error("Unknown Fi ID");
     }
     // Invoke the creation function
     return (i->second)();
  };

private:
  CallbackMap callbacks_;
};

class OptimizatorFactory
{
public:
  typedef Optimizator* (*CreateOptimizatorCallback)();
private:
  typedef map<int, CreateOptimizatorCallback> CallbackMap;
public:
  // Returns true if registration was succesfull
  bool RegisterOptimizator(int OptimizatorId, CreateOptimizatorCallback CreateFn)
  {
    return callbacks_.insert(CallbackMap::value_type(OptimizatorId, CreateFn)).second;
  };

  bool UnregisterOptimizator(int OptimizatorId)
  {
    return callbacks_.erase(OptimizatorId) == 1;
  };

  Optimizator* CreateOptimizator(int OptimizatorId)
  {
    CallbackMap::const_iterator i = callbacks_.find(OptimizatorId);
     if (i == callbacks_.end())
     {
     // not found
     throw std::runtime_error("Unknown Optimizator ID");
     }
     // Invoke the creation function
     return (i->second)();
  };

private:
  CallbackMap callbacks_;
};

class FilterFactory
{
public:
  typedef Filter* (*CreateFilterCallback)();
private:
  typedef map<int, CreateFilterCallback> CallbackMap;
public:
  // Returns true if registration was succesfull
  bool RegisterFilter(int FilterId, CreateFilterCallback CreateFn)
  {
    return callbacks_.insert(CallbackMap::value_type(FilterId, CreateFn)).second;
  };

  bool UnregisterFilter(int FilterId)
  {
    return callbacks_.erase(FilterId) == 1;
  };

  Filter* CreateFilter(int FilterId)
  {
    CallbackMap::const_iterator i = callbacks_.find(FilterId);
     if (i == callbacks_.end())
     {
     // not found
     throw std::runtime_error("Unknown Filter ID");
     }
     // Invoke the creation function
     return (i->second)();
  };

private:
  CallbackMap callbacks_;
};

class ObjectiveFunctionFactory
{
public:
  typedef ObjectiveFunction* (*CreateObjectiveFunctionCallback)();
private:
  typedef map<int, CreateObjectiveFunctionCallback> CallbackMap;
public:
  // Returns true if registration was succesfull
  bool RegisterObjectiveFunction(int ObjectiveFunctionId, CreateObjectiveFunctionCallback CreateFn)
  {
    return callbacks_.insert(CallbackMap::value_type(ObjectiveFunctionId, CreateFn)).second;
  };

  bool UnregisterObjectiveFunction(int ObjectiveFunctionId)
  {
    return callbacks_.erase(ObjectiveFunctionId) == 1;
  };

  ObjectiveFunction* CreateObjectiveFunction(int ObjectiveFunctionId)
  {
    CallbackMap::const_iterator i = callbacks_.find(ObjectiveFunctionId);
     if (i == callbacks_.end())
     {
     // not found
     throw std::runtime_error("Unknown ObjectiveFunction ID");
     }
     // Invoke the creation function
     return (i->second)();
  };

private:
  CallbackMap callbacks_;
};

class IoFactory
{
public:
  typedef Io* (*CreateIoCallback)();
private:
  typedef map<int, CreateIoCallback> CallbackMap;
public:
  // Returns true if registration was succesfull
  bool RegisterIo(int IoId, CreateIoCallback CreateFn)
  {
    return callbacks_.insert(CallbackMap::value_type(IoId, CreateFn)).second;
  };

  bool UnregisterIo(int IoId)
  {
    return callbacks_.erase(IoId) == 1;
  };

  Io* CreateIo(int IoId)
  {
    CallbackMap::const_iterator i = callbacks_.find(IoId);
     if (i == callbacks_.end())
     {
     // not found
     throw std::runtime_error("Unknown Io ID");
     }
     // Invoke the creation function
     return (i->second)();
  };

private:
  CallbackMap callbacks_;
};

class ErrorFactory
{
public:
  typedef Error* (*CreateErrorCallback)();
private:
  typedef map<int, CreateErrorCallback> CallbackMap;
public:
  // Returns true if registration was succesfull
  bool RegisterError(int ErrorId, CreateErrorCallback CreateFn)
  {
    return callbacks_.insert(CallbackMap::value_type(ErrorId, CreateFn)).second;
  };

  bool UnregisterError(int ErrorId)
  {
    return callbacks_.erase(ErrorId) == 1;
  };

  Error* CreateError(int ErrorId)
  {
    CallbackMap::const_iterator i = callbacks_.find(ErrorId);
     if (i == callbacks_.end())
     {
     // not found
     throw std::runtime_error("Unknown Error ID");
     }
     // Invoke the creation function
     return (i->second)();
  };

private:
  CallbackMap callbacks_;
};

namespace {
  ObjectiveFunction* CreateObjectiveFunction()
  {
    return new ObjectiveFunction;
  }
  const int ObjectiveFunctionId = 0;
  const bool Registered = Singleton<ObjectiveFunctionFactory>::Instance().RegisterObjectiveFunction(ObjectiveFunctionId, CreateObjectiveFunction);
};

#endif
