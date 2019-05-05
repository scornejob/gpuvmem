/* -------------------------------------------------------------------------
   Copyright (C) 2016-2017  Miguel Carcamo, Pablo Roman, Simon Casassus,
   Victor Moral, Fernando Rannou - miguel.carcamo@usach.cl

   This program includes Numerical Recipes (NR) based routines whose
   copyright is held by the NR authors. If NR routines are included,
   you are required to comply with the licensing set forth there.

   Part of the program also relies on an an ANSI C library for multi-stream
   random number generation from the related Prentice-Hall textbook
   Discrete-Event Simulation: A First Course by Steve Park and Larry Leemis,
   for more information please contact leemis@math.wm.edu

   Additionally, this program uses some NVIDIA routines whose copyright is held
   by NVIDIA end user license agreement (EULA).

   For the original parts of this code, the following license applies:

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program. If not, see <http://www.gnu.org/licenses/>.
 * -------------------------------------------------------------------------
 */
#include "functions.cuh"

namespace cg = cooperative_groups;

extern long M, N;
extern int numVisibilities, iterations, iterthreadsVectorNN, blocksVectorNN, nopositivity, crpix1, crpix2, image_count, \
           status_mod_in, flag_opt, verbose_flag, clip_flag, num_gpus, selected, iter, t_telescope, multigpu, firstgpu, reg_term, apply_noise, print_images, gridding;

extern cufftHandle plan1GPU;
extern cufftComplex *device_V, *device_fg_image, *device_image;
extern float *device_I;
extern Telescope *telescope;

extern float *device_dphi, *device_chi2, *device_dchi2, *device_S, *device_dchi2_total, *device_dS, *device_noise_image;
extern float noise_jypix, fg_scale, DELTAX, DELTAY, deltau, deltav, noise_cut, MINPIX, \
             minpix, lambda, ftol, random_probability, final_chi2, final_S, eta;

extern dim3 threadsPerBlockNN, numBlocksNN;

extern float beam_noise, beam_bmaj, beam_bmin, b_noise_aux, antenna_diameter, pb_factor, pb_cutoff;
extern float *initial_values, *penalizators;
extern double ra, dec;
extern float threshold;
extern float nu_0;
extern int nPenalizators, print_errors;
extern freqData data;

extern char* mempath, *out_image;

extern fitsfile *mod_in;

extern Field *fields;

extern VariablesPerField *vars_per_field;

extern varsPerGPU *vars_gpu;


// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template<class T>
struct SharedMemory
{
        __device__ inline operator       T *()
        {
                extern __shared__ int __smem[];
                return (T *)__smem;
        }

        __device__ inline operator const T *() const
        {
                extern __shared__ int __smem[];
                return (T *)__smem;
        }
};

// specialize for double to avoid unaligned memory
// access compile errors
template<>
struct SharedMemory<double>
{
        __device__ inline operator       double *()
        {
                extern __shared__ double __smem_d[];
                return (double *)__smem_d;
        }

        __device__ inline operator const double *() const
        {
                extern __shared__ double __smem_d[];
                return (double *)__smem_d;
        }
};

__host__ void goToError()
{
        if(num_gpus > 1) {
                for(int i=firstgpu+1; i<firstgpu + num_gpus; i++) {
                        cudaSetDevice(firstgpu);
                        cudaDeviceDisablePeerAccess(i);
                        cudaSetDevice(i);
                        cudaDeviceDisablePeerAccess(firstgpu);
                }

                for(int i=0; i<num_gpus; i++ ) {
                        cudaSetDevice((i%num_gpus) + firstgpu);
                        cudaDeviceReset();
                }
        }

        printf("An error has ocurred, exiting\n");
        exit(0);

}

__host__ void init_beam(int telescope)
{
        switch(telescope) {
        case 1:
                antenna_diameter = 1.4; /* CBI2 Antenna Diameter */
                pb_factor = 1.22; /* FWHM Factor */
                pb_cutoff = 90.0*RPARCM; /* radians */
                break;
        case 2:
                antenna_diameter = 12.0; /* ALMA Antenna Diameter */
                pb_factor = 1.13; /* FWHM Factor */
                pb_cutoff = 1.0*RPARCM; /* radians */
                break;
        case 3:
                antenna_diameter = 22.0; /* ATCA Antenna Diameter */
                pb_factor = 1.22; /* FWHM Factor */
                pb_cutoff = 1.0*RPARCM; /* radians */
                break;
        case 4:
                antenna_diameter = 25.0; /* VLA Antenna Diameter */
                pb_factor = 1.22; /* FWHM Factor */
                pb_cutoff = 20.0*RPARCM; /* radians */
                break;
        case 5:
                antenna_diameter = 3.5; /* SZA Antenna Diameter */
                pb_factor = 1.22; /* FWHM Factor */
                pb_cutoff = 20.0*RPARCM; /* radians */
                break;
        case 6:
                antenna_diameter = 0.9; /* CBI Antenna Diameter */
                pb_factor = 1.22; /* FWHM Factor */
                pb_cutoff = 80.0*RPARCM; /* radians */
                break;
        default:
                printf("Telescope type not defined\n");
                goToError();
                break;
        }
}


__host__ long NearestPowerOf2(long x)
{
        --x;
        x |= x >> 1;
        x |= x >> 2;
        x |= x >> 4;
        x |= x >> 8;
        x |= x >> 16;
        return ++x;
}


bool isPow2(unsigned int x)
{
        return ((x&(x-1))==0);
}


__host__ void readInputDat(char *file)
{
        FILE *fp;
        char item[50];
        float status;
        if((fp = fopen(file, "r")) == NULL) {
                printf("ERROR. The input file wasn't provided by the user.\n");
                goToError();
        }else{
                while(true) {
                        int ret = fscanf(fp, "%s %e", item, &status);

                        if(ret==EOF) {
                                break;
                        }else{
                                if (strcmp(item,"noise_cut")==0) {
                                        if(noise_cut == -1) {
                                                noise_cut = status;
                                        }
                                }else if (strcmp(item,"t_telescope")==0) {
                                        t_telescope = status;
                                }else if(strcmp(item,"ftol")==0) {
                                        ftol = status;
                                } else if(strcmp(item,"random_probability")==0) {
                                        if(random_probability == -1) {
                                                random_probability = status;
                                        }
                                }else{
                                        printf("Keyword not defined in input\n");
                                        goToError();
                                }
                        }
                }
        }
}

__host__ void print_help() {
        printf("Example: ./bin/gpuvmem options [ arguments ...]\n");
        printf("    -h  --help             Shows this\n");
        printf( "   -X  --blockSizeX       Block X Size for Image (Needs to be pow of 2)\n");
        printf( "   -Y  --blockSizeY       Block Y Size for Image (Needs to be pow of 2)\n");
        printf( "   -V  --blockSizeV       Block Size for Visibilities (Needs to be pow of 2)\n");
        printf( "   -i  --input            The name of the input file of visibilities(MS)\n");
        printf( "   -o  --output           The name of the output file of residual visibilities(MS)\n");
        printf( "   -O  --output-image     The name of the output image FITS file\n");
        printf("    -I  --inputdat         The name of the input file of parameters\n");
        printf("    -m  --modin            mod_in_0 FITS file\n");
        printf("    -n  --noise            Noise Parameter (Optional)\n");
        printf("    -N  --noise-cut        Noise-cut Parameter (Optional)\n");
        printf("    -r  --randoms          Percentage of data used when random sampling (Default = 1.0, optional)\n");
        printf("    -P  --prior            Prior used to regularize the solution (Default = 0 = Entropy)\n");
        printf("    -e  --eta              Variable that controls the minimum image value (Default eta = -1.0)\n");
        printf("    -p  --path             MEM path to save FITS images. With last / included. (Example ./../mem/)\n");
        printf("    -f  --file             Output file where final objective function values are saved (Optional)\n");
        printf("    -M  --multigpu         Number of GPUs to use multiGPU image synthesis (Default OFF => 0)\n");
        printf("    -s  --select           If multigpu option is OFF, then select the GPU ID of the GPU you will work on. (Default = 0)\n");
        printf("    -t  --iterations       Number of iterations for optimization (Default = 500)\n");
        printf("    -g  --gridding         Use gridding to decrease the number of visibilities. This is done in CPU (Need to select the CPU threads that will grid the input visibilities)\n");
        printf("    -z  --initial-cond     Initial conditions for image/s\n");
        printf("    -Z  --penalizators     penalizators for Fi\n");
        printf("    -c  --copyright        Shows copyright conditions\n");
        printf("    -w  --warranty         Shows no warranty details\n");
        printf("        --xcorr            Run gpuvmem with cross-correlation\n");
        printf("        --nopositivity     Run gpuvmem using chi2 with no posititivy restriction\n");
        printf("        --apply-noise      Apply random gaussian noise to visibilities\n");
        printf("        --clipping         Clips the image to positive values\n");
        printf("        --print-images     Prints images per iteration\n");
        printf("        --print-errors     Prints final error images\n");
        printf("        --verbose          Shows information through all the execution\n");
}

__host__ char *strip(const char *string, const char *chars)
{
        char * newstr = (char*)malloc(strlen(string) + 1);
        int counter = 0;

        for (; *string; string++) {
                if (!strchr(chars, *string)) {
                        newstr[ counter ] = *string;
                        ++counter;
                }
        }

        newstr[counter] = 0;
        return newstr;
}

__host__ Vars getOptions(int argc, char **argv) {
        Vars variables;
        variables.multigpu = "NULL";
        variables.ofile = "NULL";
        variables.path = "mem/";
        variables.output_image = "mod_out.fits";
        variables.select = 0;
        variables.initial_values = "NULL";
        variables.penalization_factors = "NULL";
        variables.blockSizeX = -1;
        variables.blockSizeY = -1;
        variables.blockSizeV = -1;
        variables.it_max = 500;
        variables.noise = -1;
        variables.randoms = 1.0;
        variables.noise_cut = -1;
        variables.reg_term = 0;
        variables.eta = -1.0;
        variables.gridding = 0;
        variables.nu_0 = -1;
        variables.threshold = 0.0;


        long next_op;
        const char* const short_op = "hcwi:o:O:I:m:n:N:r:f:M:s:e:p:P:X:Y:V:t:g:z:T:F:Z:";

        const struct option long_op[] = { //Flag for help, copyright and warranty
                {"help", 0, NULL, 'h' },
                {"warranty", 0, NULL, 'w' },
                {"copyright", 0, NULL, 'c' },
                /* These options set a flag. */
                {"verbose", 0, &verbose_flag, 1},
                {"nopositivity", 0, &nopositivity, 1},
                {"clipping", 0, &clip_flag, 1},
                {"apply-noise", 0, &apply_noise, 1},
                {"print-images", 0, &print_images, 1},
                {"print-errors", 0, &print_errors, 1},
                /* These options don’t set a flag. */
                {"input", 1, NULL, 'i' }, {"output", 1, NULL, 'o'}, {"output-image", 1, NULL, 'O'},
                {"threshold", 0, NULL, 'T'}, {"nu_0", 0, NULL, 'F'},
                {"inputdat", 1, NULL, 'I'}, {"modin", 1, NULL, 'm' }, {"noise", 0, NULL, 'n' },
                {"multigpu", 0, NULL, 'M'}, {"select", 1, NULL, 's'},
                {"path", 1, NULL, 'p'}, {"prior", 0, NULL, 'P'}, {"eta", 0, NULL, 'e'},
                {"blockSizeX", 1, NULL, 'X'}, {"blockSizeY", 1, NULL, 'Y'}, {"blockSizeV", 1, NULL, 'V'},
                {"iterations", 0, NULL, 't'}, {"noise-cut", 0, NULL, 'N' }, {"initial-cond", 1, NULL, 'z'}, {"penalizators", 0, NULL, 'Z'},
                {"randoms", 0, NULL, 'r' }, {"file", 0, NULL, 'f' }, {"gridding", 0, NULL, 'g' }, { NULL, 0, NULL, 0 }
        };

        if (argc == 1) {
                printf(
                        "ERROR. THE PROGRAM HAS BEEN EXECUTED WITHOUT THE NEEDED PARAMETERS OR OPTIONS\n");
                print_help();
                exit(EXIT_SUCCESS);
        }
        int option_index = 0;
        while (1) {
                next_op = getopt_long(argc, argv, short_op, long_op, &option_index);
                if (next_op == -1) {
                        break;
                }

                switch (next_op) {
                case 0:
                        /* If this option set a flag, do nothing else now. */
                        if (long_op[option_index].flag != 0)
                                break;
                        printf ("option %s", long_op[option_index].name);
                        if (optarg)
                                printf (" with arg %s", optarg);
                        printf ("\n");
                        break;
                case 'h':
                        print_help();
                        exit(EXIT_SUCCESS);
                case 'w':
                        print_warranty();
                        exit(EXIT_SUCCESS);
                case 'c':
                        print_copyright();
                        exit(EXIT_SUCCESS);
                case 'i':
                        variables.input = (char*) malloc((strlen(optarg)+1)*sizeof(char));
                        strcpy(variables.input, optarg);
                        break;
                case 'o':
                        variables.output = (char*) malloc((strlen(optarg)+1)*sizeof(char));
                        strcpy(variables.output, optarg);
                        break;
                case 'O':
                        variables.output_image = (char*) malloc((strlen(optarg)+1)*sizeof(char));
                        strcpy(variables.output_image, optarg);
                        break;
                case 'I':
                        variables.inputdat = (char*) malloc((strlen(optarg)+1)*sizeof(char));
                        strcpy(variables.inputdat, optarg);
                        break;
                case 'm':
                        variables.modin = (char*) malloc((strlen(optarg)+1)*sizeof(char));
                        strcpy(variables.modin, optarg);
                        break;
                case 'n':
                        variables.noise = atof(optarg);
                        break;
                case 'e':
                        variables.eta = atof(optarg);
                        break;
                case 'N':
                        variables.noise_cut = atof(optarg);
                        break;
                case 'F':
                        variables.nu_0 = atof(optarg);
                        break;
                case 'T':
                        variables.threshold = atof(optarg);
                        break;
                case 'p':
                        variables.path = (char*) malloc((strlen(optarg)+1)*sizeof(char));
                        strcpy(variables.path, optarg);
                        break;
                case 'P':
                        variables.reg_term = atoi(optarg);;
                        break;
                case 'M':
                        variables.multigpu = optarg;
                        break;
                case 'r':
                        variables.randoms = atof(optarg);
                        break;
                case 'f':
                        variables.ofile = (char*) malloc((strlen(optarg)+1)*sizeof(char));
                        strcpy(variables.ofile, optarg);
                        break;
                case 's':
                        variables.select = atoi(optarg);
                        break;
                case 'X':
                        variables.blockSizeX = atoi(optarg);
                        break;
                case 'Y':
                        variables.blockSizeY = atoi(optarg);
                        break;
                case 'V':
                        variables.blockSizeV = atoi(optarg);
                        break;
                case 't':
                        variables.it_max = atoi(optarg);
                        break;
                case 'g':
                        variables.gridding = atoi(optarg);
                        break;
                case 'z':
                        variables.initial_values = (char*) malloc((strlen(optarg)+1)*sizeof(char));
                        strcpy(variables.initial_values, optarg);
                        break;
                case 'Z':
                        variables.penalization_factors = (char*) malloc((strlen(optarg)+1)*sizeof(char));
                        strcpy(variables.penalization_factors, optarg);
                        break;
                case '?':
                        print_help();
                        exit(EXIT_FAILURE);
                case -1:
                        break;
                default:
                        print_help();
                        exit(EXIT_FAILURE);
                }
        }

        if(variables.blockSizeX == -1 && variables.blockSizeY == -1 && variables.blockSizeV == -1 ||
           strcmp(strip(variables.input, " "),"") == 0 && strcmp(strip(variables.output, " "),"") == 0 && strcmp(strip(variables.output_image, " "),"") == 0 && strcmp(strip(variables.inputdat, " "),"") == 0 ||
           strcmp(strip(variables.modin, " "),"") == 0 && strcmp(strip(variables.path, " "),"") == 0) {
                print_help();
                exit(EXIT_FAILURE);
        }

        if(!isPow2(variables.blockSizeX) && !isPow2(variables.blockSizeY) && !isPow2(variables.blockSizeV)) {
                print_help();
                exit(EXIT_FAILURE);
        }

        if(variables.reg_term > 3) {
                print_help();
                exit(EXIT_FAILURE);
        }

        if(variables.randoms > 1.0) {
                print_help();
                exit(EXIT_FAILURE);
        }

        if(variables.gridding < 0) {
                print_help();
                exit(EXIT_FAILURE);
        }

        if(strcmp(variables.multigpu,"NULL")!=0 && variables.select != 0) {
                print_help();
                exit(EXIT_FAILURE);
        }
        return variables;
}



#ifndef MIN
#define MIN(x,y) ((x < y) ? x : y)
#endif

__host__ void getNumBlocksAndThreads(int n, int maxBlocks, int maxThreads, int &blocks, int &threads)
{

        //get device capability, to avoid block/grid size exceed the upper bound
        cudaDeviceProp prop;
        int device;
        gpuErrchk(cudaGetDevice(&device));
        gpuErrchk(cudaGetDeviceProperties(&prop, device));


        threads = (n < maxThreads*2) ? NearestPowerOf2((n + 1)/ 2) : maxThreads;
        blocks = (n + (threads * 2 - 1)) / (threads * 2);

        if (blocks > prop.maxGridSize[0])
        {
                printf("Grid size <%d> exceeds the device capability <%d>, set block size as %d (original %d)\n",
                       blocks, prop.maxGridSize[0], threads*2, threads);

                blocks /= 2;
                threads *= 2;
        }

        blocks = MIN(maxBlocks, blocks);
}

template <class T, unsigned int blockSize, bool nIsPow2>
__global__ void deviceReduceKernel(T *g_idata, T *g_odata, unsigned int n)
{
        // Handle to thread block group
        cg::thread_block cta = cg::this_thread_block();
        T *sdata = SharedMemory<T>();

        // perform first level of reduction,
        // reading from global memory, writing to shared memory
        unsigned int tid = threadIdx.x;
        unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
        unsigned int gridSize = blockSize*2*gridDim.x;

        T mySum = 0;

        // we reduce multiple elements per thread.  The number is determined by the
        // number of active thread blocks (via gridDim).  More blocks will result
        // in a larger gridSize and therefore fewer elements per thread
        while (i < n)
        {
                mySum += g_idata[i];

                // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
                if (nIsPow2 || i + blockSize < n)
                        mySum += g_idata[i+blockSize];

                i += gridSize;
        }

        // each thread puts its local sum into shared memory
        sdata[tid] = mySum;
        cg::sync(cta);


        // do reduction in shared mem
        if ((blockSize >= 512) && (tid < 256))
        {
                sdata[tid] = mySum = mySum + sdata[tid + 256];
        }

        cg::sync(cta);

        if ((blockSize >= 256) &&(tid < 128))
        {
                sdata[tid] = mySum = mySum + sdata[tid + 128];
        }

        cg::sync(cta);

        if ((blockSize >= 128) && (tid <  64))
        {
                sdata[tid] = mySum = mySum + sdata[tid +  64];
        }

        cg::sync(cta);

#if (__CUDA_ARCH__ >= 300 )
        if ( tid < 32 )
        {
                cg::coalesced_group active = cg::coalesced_threads();

                // Fetch final intermediate sum from 2nd warp
                if (blockSize >=  64) mySum += sdata[tid + 32];
                // Reduce final warp using shuffle
                for (int offset = warpSize/2; offset > 0; offset /= 2)
                {
                        mySum += active.shfl_down(mySum, offset);
                }
        }
#else
        // fully unroll reduction within a single warp
        if ((blockSize >=  64) && (tid < 32))
        {
                sdata[tid] = mySum = mySum + sdata[tid + 32];
        }

        cg::sync(cta);

        if ((blockSize >=  32) && (tid < 16))
        {
                sdata[tid] = mySum = mySum + sdata[tid + 16];
        }

        cg::sync(cta);

        if ((blockSize >=  16) && (tid <  8))
        {
                sdata[tid] = mySum = mySum + sdata[tid +  8];
        }

        cg::sync(cta);

        if ((blockSize >=   8) && (tid <  4))
        {
                sdata[tid] = mySum = mySum + sdata[tid +  4];
        }

        cg::sync(cta);

        if ((blockSize >=   4) && (tid <  2))
        {
                sdata[tid] = mySum = mySum + sdata[tid +  2];
        }

        cg::sync(cta);

        if ((blockSize >=   2) && ( tid <  1))
        {
                sdata[tid] = mySum = mySum + sdata[tid +  1];
        }

        cg::sync(cta);
#endif

        // write result for this block to global mem
        if (tid == 0) g_odata[blockIdx.x] = mySum;
}





template <class T>
__host__ T deviceReduce(T *in, long N)
{
        T *device_out;

        int maxThreads = 256;
        int maxBlocks = NearestPowerOf2(N)/maxThreads;

        int threads = 0;
        int blocks = 0;

        getNumBlocksAndThreads(N, maxBlocks, maxThreads, blocks, threads);

        //printf("N %d, threads: %d, blocks %d\n", N, threads, blocks);
        //threads = maxThreads;
        //blocks = NearestPowerOf2(N)/threads;

        gpuErrchk(cudaMalloc(&device_out, sizeof(T)*blocks));
        gpuErrchk(cudaMemset(device_out, 0, sizeof(T)*blocks));

        int smemSize = (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);

        bool isPower2 = isPow2(N);

        dim3 dimBlock(threads, 1, 1);
        dim3 dimGrid(blocks, 1, 1);

        if(isPower2) {
                switch (threads) {
                case 512:
                        deviceReduceKernel<T, 512, true><<<dimGrid, dimBlock, smemSize>>>(in, device_out, N);
                        gpuErrchk(cudaDeviceSynchronize());
                        break;
                case 256:
                        deviceReduceKernel<T, 256, true><<<dimGrid, dimBlock, smemSize>>>(in, device_out, N);
                        gpuErrchk(cudaDeviceSynchronize());
                        break;
                case 128:
                        deviceReduceKernel<T, 128, true><<<dimGrid, dimBlock, smemSize>>>(in, device_out, N);
                        gpuErrchk(cudaDeviceSynchronize());
                        break;
                case 64:
                        deviceReduceKernel<T, 64, true><<<dimGrid, dimBlock, smemSize>>>(in, device_out, N);
                        gpuErrchk(cudaDeviceSynchronize());
                        break;
                case 32:
                        deviceReduceKernel<T, 32, true><<<dimGrid, dimBlock, smemSize>>>(in, device_out, N);
                        gpuErrchk(cudaDeviceSynchronize());
                        break;
                case 16:
                        deviceReduceKernel<T, 16, true><<<dimGrid, dimBlock, smemSize>>>(in, device_out, N);
                        gpuErrchk(cudaDeviceSynchronize());
                        break;
                case 8:
                        deviceReduceKernel<T, 8, true><<<dimGrid, dimBlock, smemSize>>>(in, device_out, N);
                        gpuErrchk(cudaDeviceSynchronize());
                        break;
                case 4:
                        deviceReduceKernel<T, 4, true><<<dimGrid, dimBlock, smemSize>>>(in, device_out, N);
                        gpuErrchk(cudaDeviceSynchronize());
                        break;
                case 2:
                        deviceReduceKernel<T, 2, true><<<dimGrid, dimBlock, smemSize>>>(in, device_out, N);
                        gpuErrchk(cudaDeviceSynchronize());
                        break;
                case 1:
                        deviceReduceKernel<T, 1, true><<<dimGrid, dimBlock, smemSize>>>(in, device_out, N);
                        gpuErrchk(cudaDeviceSynchronize());
                        break;
                }
        }else{
                switch (threads) {
                case 512:
                        deviceReduceKernel<T, 512, false><<<dimGrid, dimBlock, smemSize>>>(in, device_out, N);
                        gpuErrchk(cudaDeviceSynchronize());
                        break;
                case 256:
                        deviceReduceKernel<T, 256, false><<<dimGrid, dimBlock, smemSize>>>(in, device_out, N);
                        gpuErrchk(cudaDeviceSynchronize());
                        break;
                case 128:
                        deviceReduceKernel<T, 128, false><<<dimGrid, dimBlock, smemSize>>>(in, device_out, N);
                        gpuErrchk(cudaDeviceSynchronize());
                        break;
                case 64:
                        deviceReduceKernel<T, 64, false><<<dimGrid, dimBlock, smemSize>>>(in, device_out, N);
                        gpuErrchk(cudaDeviceSynchronize());
                        break;
                case 32:
                        deviceReduceKernel<T, 32, false><<<dimGrid, dimBlock, smemSize>>>(in, device_out, N);
                        gpuErrchk(cudaDeviceSynchronize());
                        break;
                case 16:
                        deviceReduceKernel<T, 16, false><<<dimGrid, dimBlock, smemSize>>>(in, device_out, N);
                        gpuErrchk(cudaDeviceSynchronize());
                        break;
                case 8:
                        deviceReduceKernel<T, 8, false><<<dimGrid, dimBlock, smemSize>>>(in, device_out, N);
                        gpuErrchk(cudaDeviceSynchronize());
                        break;
                case 4:
                        deviceReduceKernel<T, 4, false><<<dimGrid, dimBlock, smemSize>>>(in, device_out, N);
                        gpuErrchk(cudaDeviceSynchronize());
                        break;
                case 2:
                        deviceReduceKernel<T, 2, false><<<dimGrid, dimBlock, smemSize>>>(in, device_out, N);
                        gpuErrchk(cudaDeviceSynchronize());
                        break;
                case 1:
                        deviceReduceKernel<T, 1, false><<<dimGrid, dimBlock, smemSize>>>(in, device_out, N);
                        gpuErrchk(cudaDeviceSynchronize());
                        break;
                }
        }

        T *h_odata = (T *) malloc(blocks*sizeof(T));
        T sum = 0;

        gpuErrchk(cudaMemcpy(h_odata, device_out, blocks * sizeof(T),cudaMemcpyDeviceToHost));
        for (int i=0; i<blocks; i++)
        {
                sum += h_odata[i];
        }
        cudaFree(device_out);
        free(h_odata);
        return sum;
}

template <typename T, typename C>
__host__
void fftShift_2D(T* data, C* w, C* u, C* v, int M, int N)
{
        // 2D Slice & 1D Line
        int sLine = N;
        int sSlice = M * N;

        // Transformations Equations
        int sEq1 = (sSlice + sLine) / 2;
        int sEq2 = (sSlice - sLine) / 2;

        for(int i = 0; i < M; i++) {
                for(int j = 0; j < N; j++) {
                        // Thread Index (2D)
                        int xIndex =  j;
                        int yIndex = i;

                        // Thread Index Converted into 1D Index
                        int index = (yIndex * N) + xIndex;

                        T regTemp;
                        C uTemp;
                        C vTemp;
                        C wTemp;

                        if (xIndex < N / 2)
                        {
                                if (yIndex < M / 2)
                                {
                                        regTemp = data[index];
                                        uTemp = u[index];
                                        vTemp = v[index];
                                        wTemp = w[index];

                                        // First Quad
                                        data[index] = data[index + sEq1];
                                        u[index] = u[index + sEq1];
                                        v[index] = v[index + sEq1];
                                        w[index] = w[index + sEq1];

                                        // Third Quad
                                        data[index + sEq1] = regTemp;
                                        u[index + sEq1] = uTemp;
                                        v[index + sEq1] = vTemp;
                                        w[index + sEq1] = wTemp;
                                }
                        }
                        else
                        {
                                if (yIndex < M / 2)
                                {
                                        regTemp = data[index];
                                        uTemp = u[index];
                                        vTemp = v[index];
                                        wTemp = w[index];

                                        // Second Quad
                                        data[index] = data[index + sEq2];
                                        v[index] = u[index + sEq2];
                                        u[index] = v[index + sEq2];
                                        w[index] = w[index + sEq2];

                                        // Fourth Quad
                                        data[index + sEq2] = regTemp;
                                        u[index + sEq2] = uTemp;
                                        v[index + sEq2] = vTemp;
                                        w[index + sEq2] = wTemp;
                                }
                        }
                }
        }
}

__host__ void do_gridding(Field *fields, freqData *data, float deltau, float deltav, int M, int N)
{
        int local_max = 0;
        int max = 0;
        for(int f=0; f < data->nfields; f++) {
                for(int i=0; i < data->total_frequencies; i++) {
      #pragma omp parallel for schedule(dynamic)
                        for(int z=0; z < fields[f].numVisibilitiesPerFreq[i]; z++) {
                                int j, k;
                                float u, v;
                                float w;
                                cufftComplex Vo;

                                u  = fields[f].visibilities[i].u[z];
                                v =  fields[f].visibilities[i].v[z];
                                w = fields[f].visibilities[i].weight[z];
                                Vo = fields[f].visibilities[i].Vo[z];

                                //Correct scale and apply hermitian symmetry (it will be applied afterwards)
                                if(u < 0.0) {
                                        u *= -1.0;
                                        v *= -1.0;
                                        Vo.y *= -1.0;
                                }

                                u *= fields[f].visibilities[i].freq / LIGHTSPEED;
                                v *= fields[f].visibilities[i].freq / LIGHTSPEED;

                                j = roundf(u/fabsf(deltau) + N/2);
                                k = roundf(v/fabsf(deltav) + M/2);
        #pragma omp critical
                                {
                                        if(k < M && j < N) {
                                                fields[f].gridded_visibilities[i].Vo[N*k+j].x += w * Vo.x;
                                                fields[f].gridded_visibilities[i].Vo[N*k+j].y += w * Vo.y;
                                                fields[f].gridded_visibilities[i].weight[N*k+j] += w;
                                        }
                                }
                        }

                        int visCounter = 0;

      #pragma omp parallel for schedule(static,1)
                        for(int k=0; k<M; k++) {
                                for(int j=0; j<N; j++) {
                                        float deltau_meters = fabsf(deltau) * (LIGHTSPEED/fields[f].visibilities[i].freq);
                                        float deltav_meters = fabsf(deltav) * (LIGHTSPEED/fields[f].visibilities[i].freq);

                                        float u_meters = (j - (N/2)) * deltau_meters;
                                        float v_meters = (k - (M/2)) * deltav_meters;

                                        fields[f].gridded_visibilities[i].u[N*k+j] = u_meters;
                                        fields[f].gridded_visibilities[i].v[N*k+j] = v_meters;

                                        float weight = fields[f].gridded_visibilities[i].weight[N*k+j];
                                        if(weight > 0.0f) {
                                                fields[f].gridded_visibilities[i].Vo[N*k+j].x /= weight;
                                                fields[f].gridded_visibilities[i].Vo[N*k+j].y /= weight;
            #pragma omp atomic
                                                visCounter++;
                                        }else{
                                                fields[f].gridded_visibilities[i].weight[N*k+j] = 0.0f;
                                        }
                                }
                        }

                        fields[f].visibilities[i].u = (float*)realloc(fields[f].visibilities[i].u, visCounter*sizeof(float));
                        fields[f].visibilities[i].v = (float*)realloc(fields[f].visibilities[i].v, visCounter*sizeof(float));

                        fields[f].visibilities[i].Vo = (cufftComplex*)realloc(fields[f].visibilities[i].Vo, visCounter*sizeof(cufftComplex));

                        fields[f].visibilities[i].Vm = (cufftComplex*)malloc(visCounter*sizeof(cufftComplex));
                        memset(fields[f].visibilities[i].Vm, 0, visCounter*sizeof(cufftComplex));

                        fields[f].visibilities[i].weight = (float*)realloc(fields[f].visibilities[i].weight, visCounter*sizeof(float));

                        int l = 0;
                        for(int k=0; k<M; k++) {
                                for(int j=0; j<N; j++) {
                                        float weight = fields[f].gridded_visibilities[i].weight[N*k+j];
                                        if(weight > 0.0f) {
                                                fields[f].visibilities[i].u[l] = fields[f].gridded_visibilities[i].u[N*k+j];
                                                fields[f].visibilities[i].v[l] = fields[f].gridded_visibilities[i].v[N*k+j];
                                                fields[f].visibilities[i].Vo[l].x = fields[f].gridded_visibilities[i].Vo[N*k+j].x;
                                                fields[f].visibilities[i].Vo[l].y = fields[f].gridded_visibilities[i].Vo[N*k+j].y;
                                                fields[f].visibilities[i].weight[l] = fields[f].gridded_visibilities[i].weight[N*k+j];
                                                l++;
                                        }
                                }
                        }

                        free(fields[f].gridded_visibilities[i].u);
                        free(fields[f].gridded_visibilities[i].v);
                        free(fields[f].gridded_visibilities[i].Vo);
                        free(fields[f].gridded_visibilities[i].weight);

                        if(fields[f].numVisibilitiesPerFreq[i] > 0) {
                                fields[f].numVisibilitiesPerFreq[i] = visCounter;
                        }
                }

                local_max = *std::max_element(fields[f].numVisibilitiesPerFreq,fields[f].numVisibilitiesPerFreq+data->total_frequencies);
                if(local_max > max) {
                        max = local_max;
                }
        }


        data->max_number_visibilities_in_channel = max;
}


__host__ float calculateNoise(Field *fields, freqData data, int *total_visibilities, int blockSizeV)
{
        //Declaring block size and number of blocks for visibilities
        float sum_inverse_weight = 0.0;
        float sum_weights = 0.0;
        long UVpow2;

        for(int f=0; f<data.nfields; f++) {
                for(int i=0; i< data.total_frequencies; i++) {
                        //Calculating beam noise
                        for(int j=0; j< fields[f].numVisibilitiesPerFreq[i]; j++) {
                                if(fields[f].visibilities[i].weight[j] > 0.0) {
                                        sum_inverse_weight += 1/fields[f].visibilities[i].weight[j];
                                        sum_weights += fields[f].visibilities[i].weight[j];
                                }
                        }
                        *total_visibilities += fields[f].numVisibilitiesPerFreq[i];
                        fields[f].visibilities[i].numVisibilities = fields[f].numVisibilitiesPerFreq[i];
                        UVpow2 = NearestPowerOf2(fields[f].visibilities[i].numVisibilities);
                        fields[f].visibilities[i].threadsPerBlockUV = blockSizeV;
                        fields[f].visibilities[i].numBlocksUV = UVpow2/fields[f].visibilities[i].threadsPerBlockUV;
                }
        }


        if(verbose_flag) {
                float aux_noise = sqrt(sum_inverse_weight)/ *total_visibilities;
                printf("Calculated NOISE %e\n", aux_noise);
                printf("Using canvas NOISE anyway...\n");
                printf("Canvas NOISE = %e\n", beam_noise);
        }

        if(beam_noise == -1) {
                beam_noise = sqrt(sum_inverse_weight)/ *total_visibilities;
                if(verbose_flag) {
                        printf("No NOISE value detected in canvas...\n");
                        printf("Using NOISE: %e ...\n", beam_noise);
                }
        }

        return sum_weights;
}
/*__global__ void do_gridding(float *u, float *v, cufftComplex *Vo, cufftComplex *Vo_g, float *w, float *w_g, int* count, float deltau, float deltav, int visibilities, int M, int N)
   {
   int i = blockDim.x * blockIdx.x + threadIdx.x;
   if(i < visibilities)
   {
    int k, j;
    j = roundf(u[i]/deltau + M/2);
    k = roundf(v[i]/deltav + N/2);

    if (k < M && j < N)
    {

      atomicAdd(&Vo_g[N*k+j].x, Vo[i].x);
      atomicAdd(&Vo_g[N*k+j].y, Vo[i].y);
      atomicAdd(&w_g[N*k+j], (1.0/w[i]));
      atomicAdd(&count[N*k*j], 1);
    }
   }
   }

   __global__ void calculateCoordinates(float *u_g, float *v_g, float deltau, float deltav, int M, int N)
   {
   int j = blockDim.x * blockIdx.x + threadIdx.x;
   int i = blockDim.y * blockIdx.y + threadIdx.y;

   u_g[N*i+j] = j*deltau - (N/2)*deltau;
   v_g[N*i+j] = i*deltav - (N/2)*deltav;
   }

   __global__ void calculateAvgVar(cufftComplex *V_g, float *w_g, int *count, int M, int N)
   {
   int j = blockDim.x * blockIdx.x + threadIdx.x;
   int i = blockDim.y * blockIdx.y + threadIdx.y;

   int counter = count[N*i+j];
   if(counter > 0){
    V_g[N*i+j].x = V_g[N*i+j].x / counter;
    V_g[N*i+j].y = V_g[N*i+j].y / counter;
    w_g[N*i+j] = counter / w_g[N*i+j];
   }else{
    w_g[N*i+j] = 0.0;
   }
   }*/

__global__ void hermitianSymmetry(float *Ux, float *Vx, cufftComplex *Vo, float freq, int numVisibilities)
{
        int i = threadIdx.x + blockDim.x * blockIdx.x;

        if (i < numVisibilities) {
                if(Ux[i] < 0.0) {
                        Ux[i] *= -1.0;
                        Vx[i] *= -1.0;
                        Vo[i].y *= -1.0;
                }
                Ux[i] *= freq / LIGHTSPEED;
                Vx[i] *= freq / LIGHTSPEED;
        }
}


__device__ float AiryDiskBeam(float distance, float lambda, float antenna_diameter, float pb_factor)
{
        float atten;
        float r = pb_factor * lambda / antenna_diameter;
        float bessel_arg = PI*distance/(r/RZ);
        float bessel_func = j1f(bessel_arg);
        if(distance == 0.0f) {
                atten = 1.0f;
        }else{
                atten = 4.0f * (bessel_func/bessel_arg) * (bessel_func/bessel_arg);
        }
        return atten;
}

__device__ float GaussianBeam(float distance, float lambda, float antenna_diameter, float pb_factor)
{
        float fwhm = pb_factor * lambda / antenna_diameter;
        float c = 4.0*logf(2.0);
        float r = distance/fwhm;
        float atten = expf(-c*r*r);
        return atten;
}

__device__ float attenuation(float antenna_diameter, float pb_factor, float pb_cutoff, float freq, float xobs, float yobs, float DELTAX, float DELTAY)
{

        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        float atten_result, atten;

        int x0 = xobs;
        int y0 = yobs;
        float x = (j - x0) * DELTAX * RPDEG;
        float y = (i - y0) * DELTAY * RPDEG;

        float arc = sqrtf(x*x+y*y);
        float lambda = LIGHTSPEED/freq;

        atten = GaussianBeam(arc, lambda, antenna_diameter, pb_factor);

        if(arc <= pb_cutoff) {
                atten_result = atten;
        }else{
                atten_result = 0.0f;
        }

        return atten_result;
}


__global__ void total_attenuation(float *total_atten, float antenna_diameter, float pb_factor, float pb_cutoff, float freq, float xobs, float yobs, float DELTAX, float DELTAY, long N)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        float attenPerFreq = attenuation(antenna_diameter, pb_factor, pb_cutoff, freq, xobs, yobs, DELTAX, DELTAY);

        total_atten[N*i+j] += attenPerFreq;
}

__global__ void mean_attenuation(float *total_atten, int channels, long N)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        total_atten[N*i+j] /= channels;
}


__global__ void weight_image(float *weight_image, float *total_atten, float noise_jypix, long N)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        float atten = total_atten[N*i+j];
        weight_image[N*i+j] += (atten / noise_jypix) * (atten / noise_jypix);
}

__global__ void noise_image(float *noise_image, float *weight_image, float noise_jypix, long N)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        float noiseval;
        noiseval = sqrtf(1.0/weight_image[N*i+j]);
        noise_image[N*i+j] = noiseval;
}

__global__ void apply_beam2I(float antenna_diameter, float pb_factor, float pb_cutoff, cufftComplex *image, long N, float xobs, float yobs, float fg_scale, float freq, float DELTAX, float DELTAY)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        float atten = attenuation(antenna_diameter, pb_factor, pb_cutoff, freq, xobs, yobs, DELTAX, DELTAY);

        image[N*i+j].x = image[N*i+j].x * atten * fg_scale;
        //image[N*i+j].x = image[N*i+j].x * atten;
        image[N*i+j].y = 0.0;
}


/*--------------------------------------------------------------------
 * Phase rotate the visibility data in "image" to refer phase to point
 * (x,y) instead of (0,0).
 * Multiply pixel V(i,j) by exp(-2 pi i (x/ni + y/nj))
 *--------------------------------------------------------------------*/
__global__ void phase_rotate(cufftComplex *data, long M, long N, float xphs, float yphs)
{

        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        float u,v, phase, c, s, re, im;
        float du = xphs/(float)M;
        float dv = yphs/(float)N;

        if(j < M/2) {
                u = du * j;
        }else{
                u = du * (j-M);
        }

        if(i < N/2) {
                v = dv * i;
        }else{
                v = dv * (i-N);
        }

        phase = 2.0*(u+v);
    #if (__CUDA_ARCH__ >= 300 )
        sincospif(phase, &s, &c);
    #else
        c = cospif(phase);
        s = sinpif(phase);
    #endif
        re = data[N*i+j].x;
        im = data[N*i+j].y;
        data[N*i+j].x = re * c - im * s;
        data[N*i+j].y = re * s + im * c;
}


/*
 * Interpolate in the visibility array to find the visibility at (u,v);
 */
__global__ void vis_mod(cufftComplex *Vm, cufftComplex *V, float *Ux, float *Vx, float *weight, float deltau, float deltav, long numVisibilities, long N)
{
        int i = threadIdx.x + blockDim.x * blockIdx.x;
        long i1, i2, j1, j2;
        float du, dv, u, v;
        float v11, v12, v21, v22;
        float Zreal;
        float Zimag;

        if (i < numVisibilities) {

                u = Ux[i]/deltau;
                v = Vx[i]/deltav;

                if (fabsf(u) <= (N/2)+0.5 && fabsf(v) <= (N/2)+0.5) {

                        if(u < 0.0) {
                                u = N + u;
                        }

                        if(v < 0.0) {
                                v = N + v;
                        }

                        i1 = u;
                        i2 = (i1+1)%N;
                        du = u - i1;
                        j1 = v;
                        j2 = (j1+1)%N;
                        dv = v - j1;

                        if (i1 >= 0 && i1 < N && i2 >= 0 && i2 < N && j1 >= 0 && j1 < N && j2 >= 0 && j2 < N) {
                                /* Bilinear interpolation: real part */
                                v11 = V[N*j1 + i1].x; /* [i1, j1] */
                                v12 = V[N*j2 + i1].x; /* [i1, j2] */
                                v21 = V[N*j1 + i2].x; /* [i2, j1] */
                                v22 = V[N*j2 + i2].x; /* [i2, j2] */
                                Zreal = (1-du)*(1-dv)*v11 + (1-du)*dv*v12 + du*(1-dv)*v21 + du*dv*v22;
                                /* Bilinear interpolation: imaginary part */
                                v11 = V[N*j1 + i1].y; /* [i1, j1] */
                                v12 = V[N*j2 + i1].y; /* [i1, j2] */
                                v21 = V[N*j1 + i2].y; /* [i2, j1] */
                                v22 = V[N*j2 + i2].y; /* [i2, j2] */
                                Zimag = (1-du)*(1-dv)*v11 + (1-du)*dv*v12 + du*(1-dv)*v21 + du*dv*v22;

                                Vm[i].x = Zreal;
                                Vm[i].y = Zimag;
                        }else{
                                weight[i] = 0.0f;
                        }
                }else{
                        //Vm[i].x = 0.0f;
                        //Vm[i].y = 0.0f;
                        weight[i] = 0.0f;
                }

        }

}


__global__ void residual(cufftComplex *Vr, cufftComplex *Vm, cufftComplex *Vo, long numVisibilities){
        int i = threadIdx.x + blockDim.x * blockIdx.x;
        if (i < numVisibilities) {
                Vr[i].x = Vm[i].x - Vo[i].x;
                Vr[i].y = Vm[i].y - Vo[i].y;
        }
}



__global__ void clipWNoise(cufftComplex *fg_image, float *noise, float *I, long N, float noise_cut, float MINPIX, float eta)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;


        if(noise[N*i+j] > noise_cut) {
                if(eta > 0.0) {
                        I[N*i+j] = 0.0;
                }
                else{
                        I[N*i+j] = -1.0 * eta * MINPIX;
                }

        }

        fg_image[N*i+j].x = I[N*i+j];
        fg_image[N*i+j].y = 0;
}

__global__ void clip2IWNoise(float *noise, float *I, long N, long M, float noise_cut, float MINPIX, float alpha_start, float eta, float threshold, int schedule)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        if(noise[N*i+j] > noise_cut) {
                if(eta > 0.0) {
                        I[N*i+j] = 0.0;
                }
                else{
                        I[N*i+j] = -1.0 * eta * MINPIX;
                }
                I[N*M+N*i+j] = 0.0f;
        }else{
                if(I[N*i+j] < threshold && schedule > 0) {
                        I[N*M+N*i+j] = 0.0f;
                }
        }
}


__global__ void getGandDGG(float *gg, float *dgg, float *xi, float *g, long N)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        gg[N*i+j] = g[N*i+j] * g[N*i+j];
        dgg[N*i+j] = (xi[N*i+j] + g[N*i+j]) * xi[N*i+j];
}

__global__ void getGGandDGG(float *gg, float *dgg, float* xi, float* g, long N, long M, int image)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        float gg_temp;
        float dgg_temp;

        gg_temp = g[M*N*image+N*i+j] * g[M*N*image+N*i+j];

        dgg_temp = (xi[M*N*image+N*i+j] + g[M*N*image+N*i+j]) * xi[M*N*image+N*i+j];

        gg[N*i+j] += gg_temp;
        dgg[N*i+j] += dgg_temp;
}

__global__ void clip(cufftComplex *I, long N, float MINPIX)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        if(I[N*i+j].x < MINPIX && MINPIX >= 0.0) {
                I[N*i+j].x = MINPIX;
        }
        I[N*i+j].y = 0;
}

__global__ void clip(float *I, long N, float MINPIX)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        if(I[N*i+j] < MINPIX && MINPIX >= 0.0) {
                I[N*i+j] = MINPIX;
        }
}

__global__ void clip2I(float *I, long N, float MINPIX)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        if(I[N*i+j] < MINPIX && MINPIX >= 0.0) {
                I[N*i+j] = MINPIX;
        }
}

__global__ void newP(float *p, float *xi, float xmin, float MINPIX, float eta, long N)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        xi[N*i+j] *= xmin;
        if(p[N*i+j] + xi[N*i+j] > -1.0*eta*MINPIX) {
                p[N*i+j] += xi[N*i+j];
        }else{
                p[N*i+j] = -1.0*eta*MINPIX;
                xi[N*i+j] = 0.0;
        }
        //p[N*i+j].y = 0.0;
}

__global__ void newP(float*p, float*xi, float xmin, long N, long M, float MINPIX, float eta, int image)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        xi[N*M*image+N*i+j] *= xmin;

        if(p[N*M*image+N*i+j] + xi[N*M*image+N*i+j] > -1.0*eta*MINPIX) {
                p[N*M*image+N*i+j] += xi[N*M*image+N*i+j];
        }else{
                p[N*M*image+N*i+j] = -1.0*eta*MINPIX;
                xi[N*M*image+N*i+j] = 0.0;
        }
}

__global__ void newPNoPositivity(float *p, float *xi, float xmin, long N, long M, int image)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        xi[N*M*image+N*i+j] *= xmin;
        p[N*M*image+N*i+j] += xi[N*M*image+N*i+j];
}

__global__ void evaluateXt(float *xt, float *pcom, float *xicom, float x, float MINPIX, float eta, long N)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        if(pcom[N*i+j] + x * xicom[N*i+j] > -1.0*eta*MINPIX) {
                xt[N*i+j] = pcom[N*i+j] + x * xicom[N*i+j];
        }else{
                xt[N*i+j] = -1.0*eta*MINPIX;
        }
        //xt[N*i+j].y = 0.0;
}

__global__ void evaluateXt(float*xt, float*pcom, float*xicom, float x, long N, long M, float MINPIX, float eta, int image)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        if(pcom[N*M*image+N*i+j] + x * xicom[N*M*image+N*i+j] > -1.0*eta*MINPIX) {
                xt[N*M*image+N*i+j] = pcom[N*M*image+N*i+j] + x * xicom[N*M*image+N*i+j];
        }else{
                xt[N*M*image+N*i+j] = -1.0*eta*MINPIX;
        }
}


__global__ void evaluateXtNoPositivity(float *xt, float *pcom, float *xicom, float x, long N, long M, int image)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        xt[N*M*image+N*i+j] = pcom[N*M*image+N*i+j] + x * xicom[N*M*image+N*i+j];
}


__global__ void chi2Vector(float *chi2, cufftComplex *Vr, float *w, long numVisibilities)
{
        int i = threadIdx.x + blockDim.x * blockIdx.x;

        if (i < numVisibilities) {
                chi2[i] =  w[i] * ((Vr[i].x * Vr[i].x) + (Vr[i].y * Vr[i].y));
        }

}

__global__ void SVector(float *S, float *noise, float *I, long N, long M, float noise_cut, float MINPIX, float eta, int image)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        float entropy = 0.0;
        if(noise[N*i+j] <= noise_cut) {
                entropy = I[N*M*image+N*i+j] * logf((I[N*M*image+N*i+j] / MINPIX) + (eta + 1.0));
        }

        S[N*i+j] = entropy;
}

__global__ void QPVector(float *Q, float *noise, float *I, long N, long M, float noise_cut, int index)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        float qp = 0.0;
        if(noise[N*i+j] <= noise_cut) {
                if((i>0 && i<N-1) && (j>0 && j<N-1)) {
                        qp = (I[N*M*index+N*i+j] - I[N*M*index+N*i+(j-1)]) * (I[N*M*index+N*i+j] - I[N*M*index+N*i+(j-1)]) +
                             (I[N*M*index+N*i+j] - I[N*M*index+N*i+(j+1)]) * (I[N*M*index+N*i+j] - I[N*M*index+N*i+(j+1)]) +
                             (I[N*M*index+N*i+j] - I[N*M*index+N*(i-1)+j]) * (I[N*M*index+N*i+j] - I[N*M*index+N*(i-1)+j]) +
                             (I[N*M*index+N*i+j] - I[N*M*index+N*(i+1)+j]) * (I[N*M*index+N*i+j] - I[N*M*index+N*(i+1)+j]);
                        qp /= 2.0;
                }else{
                        qp = I[N*M*index+N*i+j];
                }
        }

        Q[N*i+j] = qp;
}

__global__ void TVVector(float *TV, float *noise, float *I, long N, long M, float noise_cut, int index)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        float tv = 0.0;
        if(noise[N*i+j] <= noise_cut) {
                if(i < N-1 && j < N-1) {
                        float dx = I[N*M*index+N*i+j] - I[N*M*index+N*i+(j+1)];
                        float dy = I[N*M*index+N*i+j] - I[N*M*index+N*(i+1)+j];
                        tv = sqrtf((dx * dx) + (dy * dy));
                }else{
                        tv = I[N*M*index+N*i+j];
                }
        }

        TV[N*i+j] = tv;
}

__global__ void LVector(float *L, float *noise, float *I, long N, long M, float noise_cut, int index)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        float Dx, Dy;

        if(noise[N*i+j] <= noise_cut) {
                if((i>0 && i<N-1) && (j>0 && j<N-1)) {
                        Dx = I[N*M*index+N*i+(j-1)] - 2 * I[N*M*index+N*i+j] + I[N*M*index+N*i+(j+1)];
                        Dy = I[N*M*index+N*(i-1)+j] - 2 * I[N*M*index+N*i+j] + I[N*M*index+N*(i+1)+j];
                        L[N*i+j] = 0.5 * (Dx + Dy) * (Dx + Dy);
                }else{
                        L[N*i+j] = I[N*M*index+N*i+j];
                }
        }
}

__global__ void searchDirection(float *g, float *xi, float *h, long N)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        g[N*i+j] = -xi[N*i+j];
        xi[N*i+j] = h[N*i+j] = g[N*i+j];
}

__global__ void searchDirection_LBFGS(float *xi, long N, long M, int image)
{
  int j = threadIdx.x + blockDim.x * blockIdx.x;
	int i = threadIdx.y + blockDim.y * blockIdx.y;

  xi[M*N*image+N*i+j] *= -1.0f;
}

__global__ void getDot_LBFGS_ff(float *aux_vector, float *vec_1, float *vec_2, int k, int h, int M, int N, int image)
{
  int j = threadIdx.x + blockDim.x * blockIdx.x;
	int i = threadIdx.y + blockDim.y * blockIdx.y;

  aux_vector[N*i+j] = vec_1[M*N*image*k + M*N*image + (N*i+j)]*vec_2[M*N*image*h + M*N*image + (N*i+j)];
}

__global__ void updateQ (float *d_q, float alpha, float *d_y, int k, int M, int N, int image)
{
  int j = threadIdx.x + blockDim.x * blockIdx.x;
	int i = threadIdx.y + blockDim.y * blockIdx.y;

  d_q[M*N*image+N*i+j] += alpha *d_y[M*N*image + M*N*k + (N*i+j)];
}

__global__ void getR (float *d_r, float *d_q, float scalar, int M, int N, int image)
{
  int j = threadIdx.x + blockDim.x * blockIdx.x;
	int i = threadIdx.y + blockDim.y * blockIdx.y;

  d_r[M*N*image+N*i+j] = d_q[M*N*image+N*i+j] * scalar;
}

__global__ void calculateSandY (float *d_y, float *d_s, float *p, float *xi, float *p_old, float *xi_old, int iter, int M, int N, int image)
{
  int j = threadIdx.x + blockDim.x * blockIdx.x;
	int i = threadIdx.y + blockDim.y * blockIdx.y;

  d_y[M*N*image*iter + M*N*image + (N*i+j)] = xi[M*N*image+N*i+j] - (-1.0f*xi_old[M*N*image+N*i+j]);
  d_s[M*N*image*iter + M*N*image + (N*i+j)] = p[M*N*image+N*i+j] - p_old[M*N*image+N*i+j];

}

__global__ void searchDirection(float* g, float* xi, float* h, long N, long M, int image)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        g[M*N*image+N*i+j] = -xi[M*N*image+N*i+j];

        xi[M*N*image+N*i+j] = h[M*N*image+N*i+j] = g[M*N*image+N*i+j];
}

__global__ void newXi(float *g, float *xi, float *h, float gam, long N)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        g[N*i+j] = -xi[N*i+j];
        xi[N*i+j] = h[N*i+j] = g[N*i+j] + gam * h[N*i+j];
}

__global__ void newXi(float* g, float* xi, float* h, float gam, long N, long M, int image)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        g[M*N*image+N*i+j] = -xi[M*N*image+N*i+j];

        xi[M*N*image+N*i+j] = h[M*N*image+N*i+j] = g[M*N*image+N*i+j] + gam * h[M*N*image+N*i+j];
}

__global__ void restartDPhi(float *dphi, float *dChi2, float *dH, long N)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        dphi[N*i+j] = 0.0;
        dChi2[N*i+j] = 0.0;
        dH[N*i+j] = 0.0;

}

__global__ void DS(float *dH, float *I, float *noise, float noise_cut, float lambda, float MINPIX, float eta, long N, long M, int image)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        if(noise[N*i+j] <= noise_cut) {
                if(I[N*M*image+N*i+j] != 0.0) {
                        dH[N*i+j] = lambda * (logf((I[N*M*image+N*i+j] / MINPIX) + (eta+1.0)) + 1.0/(1.0 + (((eta+1.0)*MINPIX) / I[N*M*image+N*i+j])));
                }else{
                        dH[N*i+j] = lambda * logf((I[N*M*image+N*i+j] / MINPIX));
                }
        }
}

__global__ void DQ(float *dQ, float *I, float *noise, float noise_cut, float lambda, long N, long M, int index)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        if(noise[N*i+j] <= noise_cut) {
                if((i>0 && i<N-1) && (j>0 && j<N-1)) {
                        dQ[N*i+j] = 2 * (4 * I[N*M*index+N*i+j] - (I[N*M*index+N*(i+1)+j] + I[N*M*index+N*(i-1)+j] + I[N*M*index+N*i+(j+1)] + I[N*M*index+N*i+(j-1)]));
                }else{
                        dQ[N*i+j] = I[N*M*index+N*i+j];
                }
                dQ[N*i+j] *= lambda;
        }
}

__global__ void DTV(float *dTV, float *I, float *noise, float noise_cut, float lambda, long N, long M, int index)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        float num0, num1, num2;
        float den0, den1, den2;
        float dtv;

        if(noise[N*i+j] <= noise_cut) {
                if((i>0 && i<N-1) && (j>0 && j<N-1)) {
                        num0 = 2 * I[N*M*index+N*i+j] - I[N*M*index+N*i+(j+1)] - I[N*M*index+N*(i+1)+j];
                        num1 = I[N*M*index+N*i+j] - I[N*M*index+N*i+(j-1)];
                        num2 = I[N*M*index+N*i+j] - I[N*M*index+N*(i-1)+j];

                        den0 = (I[N*M*index+N*i+j] - I[N*M*index+N*i+(j+1)]) * (I[N*M*index+N*i+j] - I[N*M*index+N*i+(j+1)]) +
                               (I[N*M*index+N*i+j] - I[N*M*index+N*(i+1)+j]) * (I[N*M*index+N*i+j] - I[N*M*index+N*(i+1)+j]);

                        den1 = (I[N*M*index+N*i+(j-1)] - I[N*M*index+N*i+j]) * (I[N*M*index+N*i+(j-1)] - I[N*M*index+N*i+j]) +
                               (I[N*M*index+N*i+(j-1)] - I[N*M*index+N*(i+1)+(j-1)]) * (I[N*M*index+N*i+(j-1)] - I[N*M*index+N*(i+1)+(j-1)]);

                        den2 = (I[N*M*index+N*(i-1)+j] - I[N*M*index+N*(i-1)+(j+1)]) * (I[N*M*index+N*(i-1)+j] - I[N*M*index+N*(i-1)+(j+1)]) +
                               (I[N*M*index+N*(i-1)+j] - I[N*M*index+N*i+j]) * (I[N*M*index+N*(i-1)+j] - I[N*M*index+N*i+j]);
                        if(den0 == 0 || den1 == 0 || den2 == 0) {
                                dtv = I[N*M*index+N*i+j];
                        }else{
                                dtv = num0/sqrtf(den0) + num1/sqrtf(den1) + num2/sqrtf(den2);
                        }
                }else{
                        dtv = I[N*M*index+N*i+j];
                }
                dTV[N*i+j] = lambda * dtv;
        }
}

__global__ void DL(float *dL, float *I, float *noise, float noise_cut, float lambda, long N, long M, int index)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        if(noise[N*i+j] <= noise_cut) {
                if((i>1 && i<N-2) && (j>1 && j<N-2)) {
                        dL[N*i+j] = 20 * I[N*M*index+N*i+j] -
                                    8 * I[N*M*index+N*(i+1)+j] - 8 * I[N*M*index+N*i+(j+1)] - 8 * I[N*M*index+N*(i-1)+j] - 8 * I[N*M*index+N*i+(j-1)] +
                                    2 * I[N*M*index+N*(i+1)+(j-1)] + 2 * I[N*M*index+N*(i+1)+(j+1)] + 2 * I[N*M*index+N*(i-1)+(j-1)] + 2 * I[N*M*index+N*(i-1)+(j+1)] +
                                    I[N*M*index+N*(i+2)+j] + I[N*M*index+N*i+(j+2)] + I[N*M*index+N*(i-2)+j] + I[N*M*index+N*i+(j-2)];

                }else{
                        dL[N*i+j] = 0.0;
                }
        }

        dL[N*i+j] *= lambda;

}


__global__ void DChi2_SharedMemory(float *noise, float *dChi2, cufftComplex *Vr, float *U, float *V, float *w, long N, long numVisibilities, float fg_scale, float noise_cut, float xobs, float yobs, float DELTAX, float DELTAY, float beam_fwhm, float beam_freq, float beam_cutoff, float freq)
{

        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        cg::thread_block cta = cg::this_thread_block();

        extern __shared__ float s_array[];

        int x0 = xobs;
        int y0 = yobs;
        float x = (j - x0) * DELTAX * RPDEG;
        float y = (i - y0) * DELTAY * RPDEG;

        float Ukv, Vkv, cosk, sink, atten;

        float *u_shared = s_array;
        float *v_shared = (float*)&u_shared[numVisibilities];
        float *w_shared = (float*)&v_shared[numVisibilities];
        cufftComplex *Vr_shared = (cufftComplex*)&w_shared[numVisibilities];
        if(threadIdx.x == 0 && threadIdx.y == 0){
          for(int v=0; v<numVisibilities; v++) {
            u_shared[v] = U[v];
            v_shared[v] = V[v];
            w_shared[v] = w[v];
            Vr_shared[v] = Vr[v];
            printf("u: %f, v:%f, weight: %f, real: %f, imag: %f\n", u_shared[v], v_shared[v], w_shared[v], Vr_shared[v].x, Vr_shared[v].y);
          }
        }
        cg::sync(cta);


        atten = attenuation(beam_fwhm, beam_freq, beam_cutoff, freq, xobs, yobs, DELTAX, DELTAY);

        float dchi2 = 0.0;
        if(noise[N*i+j] <= noise_cut) {
                for(int v=0; v<numVisibilities; v++) {
                        Ukv = x * u_shared[v];
                        Vkv = y * v_shared[v];
        #if (__CUDA_ARCH__ >= 300 )
                        sincospif(2.0*(Ukv+Vkv), &sink, &cosk);
        #else
                        cosk = cospif(2.0*(Ukv+Vkv));
                        sink = sinpif(2.0*(Ukv+Vkv));
        #endif
                        dchi2 += w_shared[v]*((Vr_shared[v].x * cosk) - (Vr_shared[v].y * sink));
                }

                dchi2 *= fg_scale * atten;
                dChi2[N*i+j] = dchi2;
        }
}


__global__ void DChi2(float *noise, float *dChi2, cufftComplex *Vr, float *U, float *V, float *w, long N, long numVisibilities, float fg_scale, float noise_cut, float xobs, float yobs, float DELTAX, float DELTAY, float beam_fwhm, float beam_freq, float beam_cutoff, float freq)
{

        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        int x0 = xobs;
        int y0 = yobs;
        float x = (j - x0) * DELTAX * RPDEG;
        float y = (i - y0) * DELTAY * RPDEG;

        float Ukv, Vkv, cosk, sink, atten;

        atten = attenuation(beam_fwhm, beam_freq, beam_cutoff, freq, xobs, yobs, DELTAX, DELTAY);

        float dchi2 = 0.0;
        if(noise[N*i+j] <= noise_cut) {
                for(int v=0; v<numVisibilities; v++) {
                        Ukv = x * U[v];
                        Vkv = y * V[v];
        #if (__CUDA_ARCH__ >= 300 )
                        sincospif(2.0*(Ukv+Vkv), &sink, &cosk);
        #else
                        cosk = cospif(2.0*(Ukv+Vkv));
                        sink = sinpif(2.0*(Ukv+Vkv));
        #endif
                        dchi2 += w[v]*((Vr[v].x * cosk) - (Vr[v].y * sink));
                }

                dchi2 *= fg_scale * atten;
                dChi2[N*i+j] = dchi2;
        }
}

__global__ void DChi2_total(float *dchi2_total, float *dchi2, long N)
{

        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        dchi2_total[N*i+j] += dchi2[N*i+j];
}

__global__ void DPhi(float *dphi, float *dchi2, float *dH, long N)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        dphi[N*i+j] = dchi2[N*i+j] + dH[N*i+j];
}

__global__ void AddToDPhi(float *dphi, float *dgi, long N, long M, int index)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        dphi[N*M*index+N*i+j] += dgi[N*i+j];
}

__global__ void substraction(float *x, cufftComplex *xc, float *gc, float lambda, long N)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        x[N*i+j] = xc[N*i+j].x - lambda*gc[N*i+j];
}

__global__ void projection(float *px, float *x, float MINPIX, long N){

        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;


        if(INFINITY < x[N*i+j]) {
                px[N*i+j] = INFINITY;
        }else{
                px[N*i+j] = x[N*i+j];
        }

        if(MINPIX > px[N*i+j]) {
                px[N*i+j] = MINPIX;
        }else{
                px[N*i+j] = px[N*i+j];
        }
}

__global__ void normVectorCalculation(float *normVector, float *gc, long N){
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        normVector[N*i+j] = gc[N*i+j] * gc[N*i+j];
}

__global__ void copyImage(cufftComplex *p, float *device_xt, long N)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        p[N*i+j].x = device_xt[N*i+j];
}

__global__ void calculateInu(cufftComplex *I_nu, float* I, float nu, float nu_0, float MINPIX, float eta, long N, long M)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        float I_nu_0, alpha, nudiv_pow_alpha, nudiv;

        nudiv = nu/nu_0;

        I_nu_0 = I[N*i+j];
        alpha = I[M*N+N*i+j];

        nudiv_pow_alpha = powf(nudiv, alpha);

        I_nu[N*i+j].x = I_nu_0 * nudiv_pow_alpha;

        if(I_nu[N*i+j].x < -1.0*eta*MINPIX) {
                I_nu[N*i+j].x = -1.0*eta*MINPIX;
        }

        I_nu[N*i+j].y = 0.0f;
}

__global__ void DChi2_total_alpha(float *noise, float *dchi2_total, float *dchi2, float *I, float nu, float nu_0, float noise_cut, float fg_scale, float threshold, long N, long M)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        float I_nu_0, alpha, dalpha, dI_nu_0;
        float nudiv = nu/nu_0;

        I_nu_0 = I[N*i+j];
        alpha = I[N*M+N*i+j];

        dI_nu_0 = powf(nudiv, alpha);
        dalpha = I_nu_0 * dI_nu_0 * fg_scale * logf(nudiv);
        /*if (i==242 & j==277)
           printf("nu : %e, dalpha : %e\n", nu, dalpha);*/

        if(noise[N*i+j] <= noise_cut) {
                dchi2_total[N*i+j] += dchi2[N*i+j] * dI_nu_0 * 0.0f;
                if(I_nu_0 > threshold) {
                        dchi2_total[N*M+N*i+j] += dchi2[N*i+j] * dalpha;
                }
                else{
                        dchi2_total[N*M+N*i+j] += 0.0f;
                }
        }
}

__global__ void DChi2_total_I_nu_0(float *noise, float *dchi2_total, float *dchi2, float *I, float nu, float nu_0, float noise_cut, float fg_scale, float threshold, long N, long M)
{

        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        float I_nu_0, alpha, dalpha, dI_nu_0;
        float nudiv = nu/nu_0;

        I_nu_0 = I[N*i+j];
        alpha = I[N*M+N*i+j];

        dI_nu_0 = powf(nudiv, alpha);
        dalpha = I_nu_0 * dI_nu_0 * fg_scale * logf(nudiv);

        /*if (i==242 & j==277)
           printf("nu : %e, dalpha : %e\n", nu, dalpha);*/

        if(noise[N*i+j] <= noise_cut) {
                dchi2_total[N*i+j] += dchi2[N*i+j] * dI_nu_0;
                if(I_nu_0 > threshold) {
                        dchi2_total[N*M+N*i+j] += dchi2[N*i+j] * dalpha * 0.0f;
                }
                else{
                        dchi2_total[N*M+N*i+j] += 0.0f;
                }
        }
}

__global__ void chainRule2I(float *chain, float *noise, float *I, float nu, float nu_0, float noise_cut, float fg_scale, long N, long M)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        float I_nu_0, alpha, dalpha, dI_nu_0;
        float nudiv = nu/nu_0;

        I_nu_0 = I[N*i+j];
        alpha = I[N*M+N*i+j];

        dI_nu_0 = powf(nudiv, alpha);
        dalpha = I_nu_0 * dI_nu_0 * fg_scale * logf(nudiv);

        chain[N*i+j] = dI_nu_0;
        chain[N*M+N*i+j] = dalpha;
}

__global__ void DChi2_2I(float *noise, float *chain, float *I, float *dchi2, float *dchi2_total, float threshold, float noise_cut, int image, long N, long M)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        if(noise[N*i+j] <= noise_cut && image)
        {
                if(I[N*i+j] > threshold) {
                        dchi2_total[N*i+j] += dchi2[N*i+j] * chain[N*M+N*i+j];
                }else{
                        dchi2_total[N*i+j] += 0.0f;
                }

        }else if(noise[N*i+j] <= noise_cut) {
                dchi2_total[N*i+j] += dchi2[N*i+j] * chain[N*i+j];
        }
}


__global__ void I_nu_0_Noise(float *noise_I, float *images, float *noise, float noise_cut, float nu, float nu_0, float *w, float antenna_diameter, float pb_factor, float pb_cutoff, float xobs, float yobs, float DELTAX, float DELTAY, float fg_scale, long numVisibilities, long N, long M)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        float alpha, nudiv, nudiv_pow_alpha, sum_noise, atten;

        atten = attenuation(antenna_diameter, pb_factor, pb_cutoff, nu, xobs, yobs, DELTAX, DELTAY);

        nudiv = nu/nu_0;
        alpha = images[N*M+N*i+j];
        nudiv_pow_alpha = powf(nudiv, alpha);

        sum_noise = 0.0f;
        if(noise[N*i+j] <= noise_cut) {
                for(int k=0; k<numVisibilities; k++) {
                        sum_noise +=  w[k];
                }
                noise_I[N*i+j] += fg_scale * fg_scale * atten * atten * sum_noise * nudiv_pow_alpha * nudiv_pow_alpha;
        }else{
                noise_I[N*i+j] = 0.0f;
        }



}


__global__ void alpha_Noise(float *noise_I, float *images, float nu, float nu_0, float *w, float *U, float *V, cufftComplex *Vr, float *noise, float noise_cut, float DELTAX, float DELTAY, float xobs, float yobs, float antenna_diameter, float pb_factor, float pb_cutoff, float fg_scale, long numVisibilities, long N, long M)
{
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        float I_nu, I_nu_0, alpha, nudiv, nudiv_pow_alpha, log_nu, Ukv, Vkv, cosk, sink, x, y, dchi2, sum_noise, atten;
        int x0, y0;

        x0 = xobs;
        y0 = yobs;
        x = (j - x0) * DELTAX * RPDEG;
        y = (i - y0) * DELTAY * RPDEG;

        atten = attenuation(antenna_diameter, pb_factor, pb_cutoff, nu, xobs, yobs, DELTAX, DELTAY);

        nudiv = nu/nu_0;
        I_nu_0 = images[N*i+j];
        alpha = images[N*M+N*i+j];
        nudiv_pow_alpha = powf(nudiv, alpha);

        I_nu = I_nu_0 * nudiv_pow_alpha;
        log_nu = logf(nudiv);

        sum_noise = 0.0f;
        if(noise[N*i+j] <= noise_cut) {
                for(int v=0; v<numVisibilities; v++) {
                        Ukv = x * U[v];
                        Vkv = y * V[v];
      #if (__CUDA_ARCH__ >= 300 )
                        sincospif(2.0*(Ukv+Vkv), &sink, &cosk);
      #else
                        cosk = cospif(2.0*(Ukv+Vkv));
                        sink = sinpif(2.0*(Ukv+Vkv));
      #endif
                        dchi2 = ((Vr[v].x * cosk) - (Vr[v].y * sink));
                        sum_noise += w[v] * (atten * I_nu * fg_scale + dchi2);
                }
                if(sum_noise <= 0)
                        noise_I[N*M+N*i+j]+= 0.0f;
                else
                        noise_I[N*M+N*i+j] += log_nu * log_nu * atten * I_nu * fg_scale * sum_noise;
        }else{
                noise_I[N*M+N*i+j] = 0.0f;
        }
}

__global__ void noise_reduction(float *noise_I, long N, long M){
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;

        if(noise_I[N*i+j] > 0.0f)
                noise_I[N*i+j] = 1/sqrt(noise_I[N*i+j]);
        else
                noise_I[N*i+j] = 0.0f;

        if(noise_I[N*M+N*i+j] > 0.0f)
                noise_I[N*M+N*i+j] = 1/sqrt(noise_I[N*M+N*i+j]);
        else
                noise_I[N*M+N*i+j] = 0.0f;
}


__host__ float chi2(float *I, VirtualImageProcessor *ip)
{
        if(num_gpus == 1) {
                cudaSetDevice(selected);
        }else{
                cudaSetDevice(firstgpu);
        }

        float resultPhi = 0.0;
        float resultchi2  = 0.0;
        float resultS  = 0.0;

        if(clip_flag) {
                ip->clip(I);
        }

        ip->clipWNoise(I);

        if(num_gpus == 1) {
                cudaSetDevice(selected);
                for(int f=0; f<data.nfields; f++) {
                        for(int i=0; i<data.total_frequencies; i++) {

                                if(fields[f].numVisibilitiesPerFreq[i] != 0) {

                                        gpuErrchk(cudaMemset(device_chi2, 0, sizeof(float)*data.max_number_visibilities_in_channel));

                                        ip->calculateInu(device_image, I, fields[f].visibilities[i].freq);

                                        ip->apply_beam(device_image, fields[f].global_xobs, fields[f].global_yobs, fields[f].visibilities[i].freq);
                                        gpuErrchk(cudaDeviceSynchronize());

                                        //FFT 2D
                                        if ((cufftExecC2C(plan1GPU, (cufftComplex*)device_image, (cufftComplex*)device_V, CUFFT_FORWARD)) != CUFFT_SUCCESS) {
                                                printf("CUFFT exec error\n");
                                                goToError();
                                        }
                                        gpuErrchk(cudaDeviceSynchronize());

                                        //PHASE_ROTATE
                                        phase_rotate<<<numBlocksNN, threadsPerBlockNN>>>(device_V, M, N, fields[f].global_xobs, fields[f].global_yobs);
                                        gpuErrchk(cudaDeviceSynchronize());

                                        //RESIDUAL CALCULATION
                                        //if(!gridding){
                                        vis_mod<<<fields[f].visibilities[i].numBlocksUV, fields[f].visibilities[i].threadsPerBlockUV>>>(fields[f].device_visibilities[i].Vm, device_V, fields[f].device_visibilities[i].u, fields[f].device_visibilities[i].v, fields[f].device_visibilities[i].weight, deltau, deltav, fields[f].numVisibilitiesPerFreq[i], N);
                                        gpuErrchk(cudaDeviceSynchronize());

                                        residual<<<fields[f].visibilities[i].numBlocksUV, fields[f].visibilities[i].threadsPerBlockUV>>>(fields[f].device_visibilities[i].Vr, fields[f].device_visibilities[i].Vm, fields[f].device_visibilities[i].Vo, fields[f].numVisibilitiesPerFreq[i]);
                                        gpuErrchk(cudaDeviceSynchronize());
                                        /*}else{
                                           residual<<<fields[f].visibilities[i].numBlocksUV, fields[f].visibilities[i].threadsPerBlockUV>>>(fields[f].device_visibilities[i].Vr, device_V, fields[f].device_visibilities[i].Vo, fields[f].numVisibilitiesPerFreq[i]);
                                           gpuErrchk(cudaDeviceSynchronize());

                                           }*/

                                        ////chi 2 VECTOR
                                        chi2Vector<<<fields[f].visibilities[i].numBlocksUV, fields[f].visibilities[i].threadsPerBlockUV>>>(device_chi2, fields[f].device_visibilities[i].Vr, fields[f].device_visibilities[i].weight, fields[f].numVisibilitiesPerFreq[i]);
                                        gpuErrchk(cudaDeviceSynchronize());

                                        //REDUCTIONS
                                        //chi2
                                        resultchi2  += deviceReduce<float>(device_chi2, fields[f].numVisibilitiesPerFreq[i]);
                                }
                        }
                }
        }else{
                for(int f=0; f<data.nfields; f++) {
                        #pragma omp parallel for schedule(static,1)
                        for (int i = 0; i < data.total_frequencies; i++)
                        {
                                float result = 0.0;
                                unsigned int j = omp_get_thread_num();
                                //unsigned int num_cpu_threads = omp_get_num_threads();
                                // set and check the CUDA device for this CPU thread
                                int gpu_id = -1;
                                cudaSetDevice((i%num_gpus) + firstgpu); // "% num_gpus" allows more CPU threads than GPU devices
                                cudaGetDevice(&gpu_id);
                                if(fields[f].numVisibilitiesPerFreq[i] != 0) {

                                        gpuErrchk(cudaMemset(vars_gpu[i%num_gpus].device_chi2, 0, sizeof(float)*data.max_number_visibilities_in_channel));

                                        ip->calculateInu(vars_gpu[i%num_gpus].device_image, I, fields[f].visibilities[i].freq);

                                        //apply_beam<<<numBlocksNN, threadsPerBlockNN>>>(beam_fwhm, beam_freq, beam_cutoff, vars_gpu[i%num_gpus].device_image, device_fg_image, N, fields[f].global_xobs, fields[f].global_yobs, fg_scale, fields[f].visibilities[i].freq, DELTAX, DELTAY);
                                        ip->apply_beam(vars_gpu[i%num_gpus].device_image, fields[f].global_xobs, fields[f].global_yobs, fields[f].visibilities[i].freq);
                                        gpuErrchk(cudaDeviceSynchronize());


                                        //FFT 2D
                                        if ((cufftExecC2C(vars_gpu[i%num_gpus].plan, (cufftComplex*)vars_gpu[i%num_gpus].device_image, (cufftComplex*)vars_gpu[i%num_gpus].device_V, CUFFT_FORWARD)) != CUFFT_SUCCESS) {
                                                printf("CUFFT exec error\n");
                                                //return -1 ;
                                                goToError();
                                        }
                                        gpuErrchk(cudaDeviceSynchronize());

                                        //PHASE_ROTATE
                                        phase_rotate<<<numBlocksNN, threadsPerBlockNN>>>(vars_gpu[i%num_gpus].device_V, M, N, fields[f].global_xobs, fields[f].global_yobs);
                                        gpuErrchk(cudaDeviceSynchronize());

                                        //RESIDUAL CALCULATION
                                        //if(!gridding){
                                        vis_mod<<<fields[f].visibilities[i].numBlocksUV, fields[f].visibilities[i].threadsPerBlockUV>>>(fields[f].device_visibilities[i].Vm, vars_gpu[i%num_gpus].device_V, fields[f].device_visibilities[i].u, fields[f].device_visibilities[i].v, fields[f].device_visibilities[i].weight, deltau, deltav, fields[f].numVisibilitiesPerFreq[i], N);
                                        gpuErrchk(cudaDeviceSynchronize());

                                        residual<<<fields[f].visibilities[i].numBlocksUV, fields[f].visibilities[i].threadsPerBlockUV>>>(fields[f].device_visibilities[i].Vr, fields[f].device_visibilities[i].Vm, fields[f].device_visibilities[i].Vo, fields[f].numVisibilitiesPerFreq[i]);
                                        gpuErrchk(cudaDeviceSynchronize());
                                        /*  }else{
                                            residual<<<fields[f].visibilities[i].numBlocksUV, fields[f].visibilities[i].threadsPerBlockUV>>>(fields[f].device_visibilities[i].Vr, vars_gpu[i%num_gpus].device_V, fields[f].device_visibilities[i].Vo, fields[f].numVisibilitiesPerFreq[i]);
                                            gpuErrchk(cudaDeviceSynchronize());
                                           }*/

                                        ////chi2 VECTOR
                                        chi2Vector<<<fields[f].visibilities[i].numBlocksUV, fields[f].visibilities[i].threadsPerBlockUV>>>(vars_gpu[i%num_gpus].device_chi2, fields[f].device_visibilities[i].Vr, fields[f].device_visibilities[i].weight, fields[f].numVisibilitiesPerFreq[i]);
                                        gpuErrchk(cudaDeviceSynchronize());


                                        result = deviceReduce<float>(vars_gpu[i%num_gpus].device_chi2, fields[f].numVisibilitiesPerFreq[i]);
                                        //REDUCTIONS
                                        //chi2
                                        #pragma omp atomic
                                        resultchi2  += result;
                                }
                        }
                }
        }
        if(num_gpus == 1) {
                cudaSetDevice(selected);
        }else{
                cudaSetDevice(firstgpu);
        }
        resultPhi = (0.5 * resultchi2);

        final_chi2 = resultchi2;
        /*printf("chi2 value = %.5f\n", resultchi2);
           printf("S value = %.5f\n", resultS);
           printf("(1/2) * chi2 value = %.5f\n", 0.5*resultchi2);
           printf("lambda * S value = %.5f\n", lambda*resultS);
           printf("Phi value = %.5f\n\n", resultPhi);*/

        return resultPhi;
};

__host__ void dchi2(float *I, float *dxi2, float *result_dchi2, VirtualImageProcessor *ip)
{
        if(num_gpus == 1) {
                cudaSetDevice(selected);
        }else{
                cudaSetDevice(firstgpu);
        }

        if(clip_flag) {
                ip->clip(I);
        }

        if(num_gpus == 1) {
                cudaSetDevice(selected);
                for(int f=0; f<data.nfields; f++) {
                        for(int i=0; i<data.total_frequencies; i++) {
                                if(fields[f].numVisibilitiesPerFreq[i] != 0) {
                                        //size_t shared_memory;
                                        //shared_memory = 3*fields[f].numVisibilitiesPerFreq[i]*sizeof(float) + fields[f].numVisibilitiesPerFreq[i]*sizeof(cufftComplex);
                                        DChi2<<<numBlocksNN, threadsPerBlockNN>>>(device_noise_image, device_dchi2, fields[f].device_visibilities[i].Vr, fields[f].device_visibilities[i].u, fields[f].device_visibilities[i].v, fields[f].device_visibilities[i].weight, N, fields[f].numVisibilitiesPerFreq[i], fg_scale, noise_cut, fields[f].global_xobs, fields[f].global_yobs, DELTAX, DELTAY, antenna_diameter, pb_factor, pb_cutoff, fields[f].visibilities[i].freq);
                                        //DChi2_SharedMemory<<<numBlocksNN, threadsPerBlockNN, shared_memory>>>(device_noise_image, device_dchi2, fields[f].device_visibilities[i].Vr, fields[f].device_visibilities[i].u, fields[f].device_visibilities[i].v, fields[f].device_visibilities[i].weight, N, fields[f].numVisibilitiesPerFreq[i], fg_scale, noise_cut, fields[f].global_xobs, fields[f].global_yobs, DELTAX, DELTAY, antenna_diameter, pb_factor, pb_cutoff, fields[f].visibilities[i].freq);
                                        gpuErrchk(cudaDeviceSynchronize());

                                        if(image_count == 1)
                                                DChi2_total<<<numBlocksNN, threadsPerBlockNN>>>(result_dchi2, device_dchi2, N);

                                        //ip->chainRule(I, fields[f].visibilities[i].freq);

                                        if(image_count == 2)
                                        {
                                                if(flag_opt%2 == 0)
                                                {
                                                        DChi2_total_I_nu_0<<<numBlocksNN, threadsPerBlockNN>>>(device_noise_image, result_dchi2, device_dchi2, I, fields[f].visibilities[i].freq, nu_0, noise_cut, fg_scale, threshold, N, M);
                                                }else{
                                                        DChi2_total_alpha<<<numBlocksNN, threadsPerBlockNN>>>(device_noise_image, result_dchi2, device_dchi2, I, fields[f].visibilities[i].freq, nu_0, noise_cut, fg_scale, threshold, N, M);
                                                }
                                                //DChi2_2I<<<numBlocksNN, threadsPerBlockNN>>>(device_noise_image, ip->chain, I, device_dchi2, result_dchi2, threshold, noise_cut, flag_opt%2, N, M);
                                                gpuErrchk(cudaDeviceSynchronize());
                                        }

                                }
                        }
                }
        }else{
                for(int f=0; f<data.nfields; f++) {
                        #pragma omp parallel for schedule(static,1)
                        for (int i = 0; i < data.total_frequencies; i++)
                        {
                                unsigned int j = omp_get_thread_num();
                                //unsigned int num_cpu_threads = omp_get_num_threads();
                                // set and check the CUDA device for this CPU thread
                                int gpu_id = -1;
                                cudaSetDevice((i%num_gpus) + firstgpu); // "% num_gpus" allows more CPU threads than GPU devices
                                cudaGetDevice(&gpu_id);
                                if(fields[f].numVisibilitiesPerFreq[i] != 0) {
                                        //size_t shared_memory;
                                        //shared_memory = 3*fields[f].numVisibilitiesPerFreq[i]*sizeof(float) + fields[f].numVisibilitiesPerFreq[i]*sizeof(cufftComplex);
                                        DChi2<<<numBlocksNN, threadsPerBlockNN>>>(device_noise_image, vars_gpu[i%num_gpus].device_dchi2, fields[f].device_visibilities[i].Vr, fields[f].device_visibilities[i].u, fields[f].device_visibilities[i].v, fields[f].device_visibilities[i].weight, N, fields[f].numVisibilitiesPerFreq[i], fg_scale, noise_cut, fields[f].global_xobs, fields[f].global_yobs, DELTAX, DELTAY, antenna_diameter, pb_factor, pb_cutoff, fields[f].visibilities[i].freq);
                                        //DChi2<<<numBlocksNN, threadsPerBlockNN, shared_memory>>>(device_noise_image, vars_gpu[i%num_gpus].device_dchi2, fields[f].device_visibilities[i].Vr, fields[f].device_visibilities[i].u, fields[f].device_visibilities[i].v, fields[f].device_visibilities[i].weight, N, fields[f].numVisibilitiesPerFreq[i], fg_scale, noise_cut, fields[f].global_xobs, fields[f].global_yobs, DELTAX, DELTAY, antenna_diameter, pb_factor, pb_cutoff, fields[f].visibilities[i].freq);
                                        gpuErrchk(cudaDeviceSynchronize());

                                        //ip->chainRule(I, fields[f].visibilities[i].freq);

                                        #pragma omp critical
                                        {
                                                if(image_count == 1) {
                                                        DChi2_total<<<numBlocksNN, threadsPerBlockNN>>>(result_dchi2, vars_gpu[i%num_gpus].device_dchi2, N);
                                                }
                                                if(flag_opt%2 == 0)
                                                {
                                                        DChi2_total_I_nu_0<<<numBlocksNN, threadsPerBlockNN>>>(device_noise_image, result_dchi2, vars_gpu[i%num_gpus].device_dchi2, I, fields[f].visibilities[i].freq, nu_0, noise_cut, fg_scale, threshold, N, M);
                                                }else{
                                                        DChi2_total_alpha<<<numBlocksNN, threadsPerBlockNN>>>(device_noise_image, result_dchi2, vars_gpu[i%num_gpus].device_dchi2, I, fields[f].visibilities[i].freq, nu_0, noise_cut, fg_scale, threshold, N, M);
                                                }
                                                gpuErrchk(cudaDeviceSynchronize());
                                        }
                                }
                        }
                }
        }

        if(num_gpus == 1) {
                cudaSetDevice(selected);
        }else{
                cudaSetDevice(firstgpu);
        }
};

__host__ float laplacian(float *I, float * ds, float penalization_factor, int mod, int order, int imageIndex)
{
        if(num_gpus == 1) {
                cudaSetDevice(selected);
        }else{
                cudaSetDevice(firstgpu);
        }

        float resultS = 0;
        if(iter > 0 && penalization_factor)
        {
                LVector<<<numBlocksNN, threadsPerBlockNN>>>(ds, device_noise_image, I, N, M, noise_cut, imageIndex);
                gpuErrchk(cudaDeviceSynchronize());
                resultS  = deviceReduce<float>(ds, M*N);
        }
        return resultS;
};

__host__ void DLaplacian(float *I, float *dgi, float penalization_factor, float mod, float order, float index)
{
        if(num_gpus == 1) {
                cudaSetDevice(selected);
        }else{
                cudaSetDevice(firstgpu);
        }
        if(iter > 0 && penalization_factor)
        {
                DL<<<numBlocksNN, threadsPerBlockNN>>>(dgi, I, device_noise_image, noise_cut, penalization_factor, N, M, index);
                gpuErrchk(cudaDeviceSynchronize());
        }
};

__host__ void linkAddToDPhi(float *dphi, float *dgi, int index)
{
        if(num_gpus == 1) {
                cudaSetDevice(selected);
        }else{
                cudaSetDevice(firstgpu);
        }
        AddToDPhi<<<numBlocksNN, threadsPerBlockNN>>>(dphi, dgi, N, M, index);
        gpuErrchk(cudaDeviceSynchronize());
};

__host__ void defaultNewP(float*p, float*xi, float xmin, int image)
{
        newPNoPositivity<<<numBlocksNN, threadsPerBlockNN>>>(p, xi, xmin, N, M, image);
};

__host__ void particularNewP(float*p, float*xi, float xmin, int image)
{
        newP<<<numBlocksNN, threadsPerBlockNN>>>(p, xi, xmin, N, M, initial_values[image], eta, image);
};

__host__ void defaultEvaluateXt(float*xt, float*pcom, float*xicom, float x, int image)
{
        evaluateXtNoPositivity<<<numBlocksNN, threadsPerBlockNN>>>(xt, pcom, xicom, x, N, M, image);
};

__host__ void particularEvaluateXt(float*xt, float*pcom, float*xicom, float x, int image)
{
        evaluateXt<<<numBlocksNN, threadsPerBlockNN>>>(xt, pcom, xicom, x, N, M, initial_values[image], eta, image);
};

__host__ void linkClipWNoise2I(float *I)
{
        clip2IWNoise<<<numBlocksNN, threadsPerBlockNN>>>(device_noise_image, I, N, M, noise_cut, initial_values[0], initial_values[1], eta, threshold, flag_opt);
        gpuErrchk(cudaDeviceSynchronize());
};

__host__ void linkApplyBeam2I(cufftComplex *image, float xobs, float yobs, float freq)
{
        apply_beam2I<<<numBlocksNN, threadsPerBlockNN>>>(antenna_diameter, pb_factor, pb_cutoff, image, N, xobs, yobs, fg_scale, freq, DELTAX, DELTAY);
        gpuErrchk(cudaDeviceSynchronize());
};

__host__ void linkCalculateInu2I(cufftComplex *image, float *I, float freq)
{
        calculateInu<<<numBlocksNN, threadsPerBlockNN>>>(image, I, freq, nu_0, initial_values[0], eta, N, M);
        gpuErrchk(cudaDeviceSynchronize());
};

__host__ void linkClip(float *I)
{
        clip<<<numBlocksNN, threadsPerBlockNN>>>(I, N, initial_values[0]);
        gpuErrchk(cudaDeviceSynchronize());
};

__host__ void linkChain2I(float *chain, float freq, float *I)
{
        chainRule2I<<<numBlocksNN, threadsPerBlockNN>>>(chain, device_noise_image, I, freq, nu_0, noise_cut, fg_scale, N, M);
        gpuErrchk(cudaDeviceSynchronize());
};

__host__ float SEntropy(float *I, float * ds, float penalization_factor, int mod, int order, int index)
{
        if(num_gpus == 1) {
                cudaSetDevice(selected);
        }else{
                cudaSetDevice(firstgpu);
        }

        float resultS = 0;
        if(iter > 0 && penalization_factor)
        {
                SVector<<<numBlocksNN, threadsPerBlockNN>>>(ds, device_noise_image, I, N, M, noise_cut, initial_values[index], eta, index);
                gpuErrchk(cudaDeviceSynchronize());
                resultS  = deviceReduce<float>(ds, M*N);
        }
        final_S = resultS;
        return resultS;
};

__host__ void DEntropy(float *I, float *dgi, float penalization_factor, int mod, int order, int index)
{
        if(num_gpus == 1) {
                cudaSetDevice(selected);
        }else{
                cudaSetDevice(firstgpu);
        }
        if(iter > 0 && penalization_factor)
        {
                DS<<<numBlocksNN, threadsPerBlockNN>>>(dgi, I, device_noise_image, noise_cut, penalization_factor, initial_values[index], eta, N, M, index);
                gpuErrchk(cudaDeviceSynchronize());
        }
};

__host__ float quadraticP(float *I, float * ds, float penalization_factor, int mod, int order, int index)
{
        if(num_gpus == 1) {
                cudaSetDevice(selected);
        }else{
                cudaSetDevice(firstgpu);
        }

        float resultS = 0;
        if(iter > 0 && penalization_factor)
        {
                QPVector<<<numBlocksNN, threadsPerBlockNN>>>(ds, device_noise_image, I, N, M, noise_cut, index);
                gpuErrchk(cudaDeviceSynchronize());
                resultS  = deviceReduce<float>(ds, M*N);
        }
        final_S = resultS;
        return resultS;
};

__host__ void DQuadraticP(float *I, float *dgi, float penalization_factor, int mod, int order, int index)
{
        if(num_gpus == 1) {
                cudaSetDevice(selected);
        }else{
                cudaSetDevice(firstgpu);
        }
        if(iter > 0 && penalization_factor)
        {
                DQ<<<numBlocksNN, threadsPerBlockNN>>>(dgi, I, device_noise_image, noise_cut, penalization_factor, N, M, index);
                gpuErrchk(cudaDeviceSynchronize());
        }
};

__host__ float totalvariation(float *I, float * ds, float penalization_factor, int mod, int order, int index)
{
        if(num_gpus == 1) {
                cudaSetDevice(selected);
        }else{
                cudaSetDevice(firstgpu);
        }

        float resultS = 0;
        if(iter > 0 && penalization_factor)
        {
                TVVector<<<numBlocksNN, threadsPerBlockNN>>>(ds, device_noise_image, I, N, M, noise_cut, index);
                gpuErrchk(cudaDeviceSynchronize());
                resultS  = deviceReduce<float>(ds, M*N);
        }
        final_S = resultS;
        return resultS;
};

__host__ void DTVariation(float *I, float *dgi, float penalization_factor, int mod, int order, int index)
{
        if(num_gpus == 1) {
                cudaSetDevice(selected);
        }else{
                cudaSetDevice(firstgpu);
        }
        if(iter > 0 && penalization_factor)
        {
                DTV<<<numBlocksNN, threadsPerBlockNN>>>(dgi, I, device_noise_image, noise_cut, penalization_factor, N, M, index);
                gpuErrchk(cudaDeviceSynchronize());
        }
};

__host__ void calculateErrors(Image *image){

        float *errors = image->getErrorImage();

        if(num_gpus == 1) {
                cudaSetDevice(selected);
        }else{
                cudaSetDevice(firstgpu);
        }

        gpuErrchk(cudaMalloc((void**)&errors, sizeof(float)*M*N*image->getImageCount()));
        gpuErrchk(cudaMemset(errors, 0, sizeof(float)*M*N*image->getImageCount()));

        if(num_gpus == 1) {
                cudaSetDevice(selected);
                for(int f=0; f<data.nfields; f++) {
                        for(int i=0; i<data.total_frequencies; i++) {
                                if(fields[f].numVisibilitiesPerFreq[i] != 0) {
                                        I_nu_0_Noise<<<numBlocksNN, threadsPerBlockNN>>>(errors, image->getImage(), device_noise_image, noise_cut, fields[f].visibilities[i].freq, nu_0, fields[f].device_visibilities[i].weight, antenna_diameter, pb_factor, pb_cutoff, fields[f].global_xobs, fields[f].global_yobs, DELTAX, DELTAY, fg_scale, fields[f].numVisibilitiesPerFreq[i], N, M);
                                        gpuErrchk(cudaDeviceSynchronize());
                                        alpha_Noise<<<numBlocksNN, threadsPerBlockNN>>>(errors, image->getImage(), fields[f].visibilities[i].freq, nu_0, fields[f].device_visibilities[i].weight, fields[f].device_visibilities[i].u, fields[f].device_visibilities[i].v, fields[f].device_visibilities[i].Vr, device_noise_image, noise_cut, DELTAX, DELTAY, fields[f].global_xobs, fields[f].global_yobs, antenna_diameter, pb_factor, pb_cutoff, fg_scale, fields[f].numVisibilitiesPerFreq[i], N, M);
                                        gpuErrchk(cudaDeviceSynchronize());
                                }
                        }
                }
        }else{
                for(int f=0; f<data.nfields; f++) {
                        #pragma omp parallel for schedule(static,1)
                        for (int i = 0; i < data.total_frequencies; i++)
                        {
                                unsigned int j = omp_get_thread_num();
                                //unsigned int num_cpu_threads = omp_get_num_threads();
                                // set and check the CUDA device for this CPU thread
                                int gpu_id = -1;
                                cudaSetDevice((i%num_gpus) + firstgpu); // "% num_gpus" allows more CPU threads than GPU devices
                                cudaGetDevice(&gpu_id);

                                if(fields[f].numVisibilitiesPerFreq[i] != 0) {

                                        #pragma omp critical
                                        {
                                                I_nu_0_Noise<<<numBlocksNN, threadsPerBlockNN>>>(errors, image->getImage(), device_noise_image, noise_cut, fields[f].visibilities[i].freq, nu_0, fields[f].device_visibilities[i].weight, antenna_diameter, pb_factor, pb_cutoff, fields[f].global_xobs, fields[f].global_yobs, DELTAX, DELTAY, fg_scale, fields[f].numVisibilitiesPerFreq[i], N, M);
                                                gpuErrchk(cudaDeviceSynchronize());
                                                alpha_Noise<<<numBlocksNN, threadsPerBlockNN>>>(errors, image->getImage(), fields[f].visibilities[i].freq, nu_0, fields[f].device_visibilities[i].weight, fields[f].device_visibilities[i].u, fields[f].device_visibilities[i].v, fields[f].device_visibilities[i].Vr, device_noise_image, noise_cut, DELTAX, DELTAY, fields[f].global_xobs, fields[f].global_yobs, antenna_diameter, pb_factor, pb_cutoff, fg_scale, fields[f].numVisibilitiesPerFreq[i], N, M);
                                                gpuErrchk(cudaDeviceSynchronize());
                                        }
                                }
                        }
                }
        }

        noise_reduction<<<numBlocksNN, threadsPerBlockNN>>>(errors, N, M);
        gpuErrchk(cudaDeviceSynchronize());

        image->setErrorImage(errors);

}
