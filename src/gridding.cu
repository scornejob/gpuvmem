#include "gridding.cuh"
#include <iostream>

using std::cout;
using std::endl;

extern float deltau, deltav;
extern long M, N;
extern int num_gpus;

Gridding::Gridding()
{
        this->threads = 1;
};

void Gridding::applyCriteria(Visibilities *v)
{
        for(int f=0; f< v->getData()->nfields; f++) {
                for(int i=0; i < v->getData()->total_frequencies; i++) {
                        v->getFields()[f].gridded_visibilities[i].u = (float*)malloc(M*N*sizeof(float));
                        v->getFields()[f].gridded_visibilities[i].v = (float*)malloc(M*N*sizeof(float));
                        v->getFields()[f].gridded_visibilities[i].weight = (float*)malloc(M*N*sizeof(float));
                        v->getFields()[f].gridded_visibilities[i].Vo = (cufftComplex*)malloc(M*N*sizeof(cufftComplex));

                        memset(v->getFields()[f].gridded_visibilities[i].u, 0, M*N*sizeof(float));
                        memset(v->getFields()[f].gridded_visibilities[i].v, 0, M*N*sizeof(float));
                        memset(v->getFields()[f].gridded_visibilities[i].weight, 0, M*N*sizeof(float));
                        memset(v->getFields()[f].gridded_visibilities[i].Vo, 0, M*N*sizeof(cufftComplex));
                }
        }
        omp_set_num_threads(threads);
        do_gridding(v->getFields(), v->getData(), deltau, deltav, M, N, v->getTotalVisibilities());
        omp_set_num_threads(num_gpus);
};
Gridding::Gridding(int threads)
{
        if(threads != 1 && threads >= 1)
                this->threads = threads;
        else if(threads != 1)
                cout << "Number of threads setted to 1" << endl;
};

void Gridding::configure(void *params)
{
        int *threads = (int*) params;
        cout << "Number of threads = " << *threads << endl;
        if(*threads != 1 && *threads >= 1)
                this->threads = *threads;
        else if(*threads != 1)
                cout << "Number of threads setted to 1" << endl;
};

namespace {
Filter* CreateGridding()
{
        return new Gridding;
}
const int GriddingId = 0;
const bool RegisteredGridding = Singleton<FilterFactory>::Instance().RegisterFilter(GriddingId, CreateGridding);
};
