#include "gridding.cuh"

extern double deltau, deltav;
extern float robust_param;
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
                    for(int s=0; i < v->getData()->nstokes; s++) {
                        v->getFields()[f].gridded_visibilities[i][s].uvw = (double3 *) malloc(M * N * sizeof(double3));
                        v->getFields()[f].gridded_visibilities[i][s].weight = (float *) malloc(M * N * sizeof(float));
                        v->getFields()[f].gridded_visibilities[i][s].Vo = (cufftComplex *) malloc(
                                M * N * sizeof(cufftComplex));

                        memset(v->getFields()[f].gridded_visibilities[i][s].uvw, 0, M * N * sizeof(double3));
                        memset(v->getFields()[f].gridded_visibilities[i][s].weight, 0, M * N * sizeof(float));
                        memset(v->getFields()[f].gridded_visibilities[i][s].S, 0, M * N * sizeof(int));
                        memset(v->getFields()[f].gridded_visibilities[i][s].Vo, 0, M * N * sizeof(cufftComplex));
                    }
                }
        }
        omp_set_num_threads(threads);
        do_gridding(v->getFields(), v->getData(), deltau, deltav, M, N, robust_param);
        omp_set_num_threads(num_gpus);
};

Gridding::Gridding(int threads)
{
        if(threads != 1 && threads >= 1)
                this->threads = threads;
        else if(threads != 1)
                printf("Number of threads set to 1\n");
};

void Gridding::configure(void *params)
{
        int *threads = (int*) params;
        printf("Number of threads = %d\n", *threads);
        if(*threads != 1 && *threads >= 1)
                this->threads = *threads;
        else if(*threads != 1)
                printf("Number of threads set to 1\n");
};

namespace {
Filter* CreateGridding()
{
        return new Gridding;
}
const int GriddingId = 0;
const bool RegisteredGridding = Singleton<FilterFactory>::Instance().RegisterFilter(GriddingId, CreateGridding);
};
