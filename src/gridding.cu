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
        double3 valzero;
        cufftComplex complexValZero;
        valzero.x = 0.0;
        valzero.y = 0.0;
        valzero.z = 0.0;

        complexValZero.x = 0.0f;
        complexValZero.y = 0.0f;
        for(int d=0; d< v->getNDatasets(); d++) {
                for(int f=0; f< v->getMSDataset()[d].data.nfields; f++) {
                        for(int i=0; i < v->getMSDataset()[d].data.total_frequencies; i++) {
                                for(int s=0; i < v->getMSDataset()[d].data.nstokes; s++) {
                                        double3 val;
                                        v->getMSDataset()[d].fields[f].gridded_visibilities[i][s].uvw.assign(M * N, valzero);
                                        v->getMSDataset()[d].fields[f].gridded_visibilities[i][s].weight.assign(M * N, 0.0f);
                                        v->getMSDataset()[d].fields[f].gridded_visibilities[i][s].Vo.assign(M * N, complexValZero);

                                }
                        }
                }
                omp_set_num_threads(threads);
                do_gridding(v->getMSDataset()[d].fields,&v->getMSDataset()[d].data, deltau, deltav, M, N, robust_param);
                omp_set_num_threads(num_gpus);
        }
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
