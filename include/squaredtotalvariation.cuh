#ifndef STVVECTOR_CUH
#define STVVECTOR_CUH

#include "framework.cuh"
#include "functions.cuh"


class SquaredTotalVariationP : public Fi
{
public:
SquaredTotalVariationP();
float calcFi(float *p);
void calcGi(float *p, float *xi);
void restartDGi();
void addToDphi(float *device_dphi);
void configure(int penalizatorIndex, int imageIndex, int imageToAdd);
void setSandDs(float *S, float *Ds);
float calculateSecondDerivate(){
};
};

#endif
