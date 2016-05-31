#include "frprmn.cuh"
#include "directioncosines.cuh"
#include <time.h>

long M;
long N;
long numVisibilities;
int iter=0;

cufftHandle plan1GPU;

cufftComplex *device_I;
cufftComplex *device_V;
cufftComplex *device_noise_image;
cufftComplex *device_fg_image;
cufftComplex *device_image;


float *device_dphi;

float *device_dchi2_total;
float *device_dH;

float *device_chi2;
float *device_H;

dim3 threadsPerBlockNN;
dim3 numBlocksNN;

int threadsVectorReduceNN;
int blocksVectorReduceNN;

float difmap_noise;

float fg_scale;
char *output;
float global_xobs;
float global_yobs;

float noise_cut;
float MINPIX;
float minpix_factor;
float lambda;
float ftol;
float random_probability;
int positivity = 1;

float DELTAX;
float DELTAY;
float deltau;
float deltav;


float beam_noise;
float beam_bmaj;
float beam_bmin;
double ra;
double dec;
double obsra;
double obsdec;
int crpix1;
int crpix2;
freqData data;
VPF *device_vars;
Vis *visibilities;
Vis *device_visibilities;

int num_gpus;
int multigpu;
int selected;

fitsfile *mod_in;
int status_mod_in;

char *mempath;
char *out_image;

int verbose_flag;

inline bool IsGPUCapableP2P(cudaDeviceProp *pProp)
{
  #ifdef _WIN32
      return (bool)(pProp->tccDriver ? true : false);
  #else
      return (bool)(pProp->major >= 2);
  #endif
}

inline bool IsAppBuiltAs64()
{
  #if defined(__x86_64) || defined(AMD64) || defined(_M_AMD64)
      return 1;
  #else
      return 0;
  #endif
}

__host__ int main(int argc, char **argv) {
  clock_t t;
  double start, end;
	////CHECK FOR AVAILABLE GPUs
	cudaGetDeviceCount(&num_gpus);

	if(num_gpus < 1){
		printf("No CUDA capable devices were detected\n");
    return 1;
	}

	if (!IsAppBuiltAs64()){
        printf("%s is only supported with on 64-bit OSs and the application must be built as a 64-bit target. Test is being waived.\n", argv[0]);
        exit(EXIT_SUCCESS);
  }


  if(verbose_flag){
  	printf("Number of host CPUs:\t%d\n", omp_get_num_procs());
    printf("Number of CUDA devices:\t%d\n", num_gpus);


  	for(int i = 0; i < num_gpus; i++){
    	cudaDeviceProp dprop;
      cudaGetDeviceProperties(&dprop, i);

      printf("> GPU%d = \"%15s\" %s capable of Peer-to-Peer (P2P)\n", i, dprop.name, (IsGPUCapableP2P(&dprop) ? "IS " : "NOT"));

      //printf("   %d: %s\n", i, dprop.name);
    }
    printf("---------------------------\n");
  }

	float noise_min = 1E32;

	Vars variables = getOptions(argc, argv);
	char *msinput = variables.input;
	char *msoutput = variables.output;
  char *inputdat = variables.inputdat;
	char *modinput = variables.modin;
	char *beaminput = variables.beam;
	multigpu = variables.multigpu;
  selected = variables.select;
  mempath = variables.path;
  out_image = variables.output_image;


  if(selected > num_gpus || selected < 0){
    printf("ERROR. THE SELECTED GPU DOESN'T EXIST\n");
    exit(-1);
  }

  readInputDat(inputdat);
	data = getFreqs(msinput);
  if(verbose_flag){
	   printf("Number of frequencies file = %d\n", data.total_frequencies);
  }

  if(multigpu < 0 || multigpu > num_gpus){
    printf("ERROR. NUMBER OF GPUS CANNOT BE NEGATIVE OR GREATER THAN THE NUMBER OF GPUS\n");
    exit(-1);
  }else{
    if(multigpu == 0){
      num_gpus = 1;
    }else{
      if(data.total_frequencies == 1){
        printf("ONLY ONE FREQUENCY. CHANGING NUMBER OF GPUS TO 1\n");
				num_gpus = 1;
      }else{
        num_gpus = multigpu;
        omp_set_num_threads(num_gpus);
      }
    }
  }

 //printf("number of FINAL host CPUs:\t%d\n", omp_get_num_procs());
 if(verbose_flag){
   printf("Number of CUDA devices and threads: \t%d\n", num_gpus);
 }

 //Check peer access if there is more than 1 GPU
  if(num_gpus > 1){
	  for(int i=1; i<num_gpus; i++){
			cudaDeviceProp dprop0, dpropX;
			cudaGetDeviceProperties(&dprop0, 0);
			cudaGetDeviceProperties(&dpropX, i);
			int canAccessPeer0_x, canAccessPeerx_0;
			cudaDeviceCanAccessPeer(&canAccessPeer0_x, 0, i);
			cudaDeviceCanAccessPeer(&canAccessPeerx_0 , i, 0);
      if(verbose_flag){
  			printf("> Peer-to-Peer (P2P) access from %s (GPU%d) -> %s (GPU%d) : %s\n", dprop0.name, 0, dpropX.name, i, canAccessPeer0_x ? "Yes" : "No");
      	printf("> Peer-to-Peer (P2P) access from %s (GPU%d) -> %s (GPU%d) : %s\n", dpropX.name, i, dprop0.name, 0, canAccessPeerx_0 ? "Yes" : "No");
      }
			if(canAccessPeer0_x == 0 || canAccessPeerx_0 == 0){
				printf("Two or more SM 2.0 class GPUs are required for %s to run.\n", argv[0]);
        printf("Support for UVA requires a GPU with SM 2.0 capabilities.\n");
        printf("Peer to Peer access is not available between GPU%d <-> GPU%d, waiving test.\n", 0, i);
        exit(EXIT_SUCCESS);
			}else{
				cudaSetDevice(0);
        if(verbose_flag){
          printf("Granting access from 0 to %d...\n", i);
        }
				cudaDeviceEnablePeerAccess(i,0);
				cudaSetDevice(i%num_gpus);
        if(verbose_flag){
          printf("Granting access from %d to 0...\n", i);
        }
				cudaDeviceEnablePeerAccess(0,0);
        if(verbose_flag){
				      printf("Checking GPU%d and GPU%d for UVA capabilities...\n", 0, 1);
        }
				const bool has_uva = (dprop0.unifiedAddressing && dpropX.unifiedAddressing);
        if(verbose_flag){
  				printf("> %s (GPU%d) supports UVA: %s\n", dprop0.name, 0, (dprop0.unifiedAddressing ? "Yes" : "No"));
      		printf("> %s (GPU%d) supports UVA: %s\n", dpropX.name, i, (dpropX.unifiedAddressing ? "Yes" : "No"));
        }
				if (has_uva){
          if(verbose_flag){
        	   printf("Both GPUs can support UVA, enabling...\n");
          }
    		}
    		else{
        	printf("At least one of the two GPUs does NOT support UVA, waiving test.\n");
        	exit(EXIT_SUCCESS);
    		}
			}
	 	}
  }


	visibilities = (Vis*)malloc(data.total_frequencies*sizeof(Vis));
	device_visibilities = (Vis*)malloc(data.total_frequencies*sizeof(Vis));
	device_vars = (VPF*)malloc(data.total_frequencies*sizeof(VPF));



  //ONLY CPU
	for(int i=0; i < data.total_frequencies; i++){
		visibilities[i].id = (int*)malloc(data.numVisibilitiesPerFreq[i]*sizeof(int));
		visibilities[i].stokes = (int*)malloc(data.numVisibilitiesPerFreq[i]*sizeof(int));
		visibilities[i].u = (float*)malloc(data.numVisibilitiesPerFreq[i]*sizeof(float));
		visibilities[i].v = (float*)malloc(data.numVisibilitiesPerFreq[i]*sizeof(float));
		visibilities[i].weight = (float*)malloc(data.numVisibilitiesPerFreq[i]*sizeof(float));
		visibilities[i].Vo = (cufftComplex*)malloc(data.numVisibilitiesPerFreq[i]*sizeof(cufftComplex));
		visibilities[i].Vr = (cufftComplex*)malloc(data.numVisibilitiesPerFreq[i]*sizeof(cufftComplex));
	}


  if(verbose_flag){
	   printf("Reading visibilities and FITS input files...\n");
  }
	readMS(msinput, modinput, beaminput, visibilities);
  if(verbose_flag){
    printf("MS File Successfully Read\n");
  }

  //Declaring block size and number of blocks for visibilities
	for(int i=0; i<data.total_frequencies; i++){
		visibilities[i].numVisibilities = data.numVisibilitiesPerFreq[i];
		long UVpow2 = NearestPowerOf2(visibilities[i].numVisibilities);
    visibilities[i].threadsPerBlockUV = variables.blockSizeV;
		visibilities[i].numBlocksUV = UVpow2/visibilities[i].threadsPerBlockUV;
  }

	if(num_gpus == 1){
    cudaSetDevice(selected);
		for(int i=0; i<data.total_frequencies; i++){
			gpuErrchk(cudaMalloc(&device_visibilities[i].u, sizeof(float)*data.numVisibilitiesPerFreq[i]));
			gpuErrchk(cudaMalloc(&device_visibilities[i].v, sizeof(float)*data.numVisibilitiesPerFreq[i]));
			gpuErrchk(cudaMalloc(&device_visibilities[i].Vo, sizeof(cufftComplex)*data.numVisibilitiesPerFreq[i]));
			gpuErrchk(cudaMalloc(&device_visibilities[i].weight, sizeof(float)*data.numVisibilitiesPerFreq[i]));
			gpuErrchk(cudaMalloc(&device_visibilities[i].Vr, sizeof(cufftComplex)*data.numVisibilitiesPerFreq[i]));
		}
	}else{
		for(int i=0; i<data.total_frequencies; i++){
			cudaSetDevice(i%num_gpus);
			gpuErrchk(cudaMalloc(&device_visibilities[i].u, sizeof(float)*data.numVisibilitiesPerFreq[i]));
			gpuErrchk(cudaMalloc(&device_visibilities[i].v, sizeof(float)*data.numVisibilitiesPerFreq[i]));
			gpuErrchk(cudaMalloc(&device_visibilities[i].Vo, sizeof(cufftComplex)*data.numVisibilitiesPerFreq[i]));
			gpuErrchk(cudaMalloc(&device_visibilities[i].weight, sizeof(float)*data.numVisibilitiesPerFreq[i]));
			gpuErrchk(cudaMalloc(&device_visibilities[i].Vr, sizeof(cufftComplex)*data.numVisibilitiesPerFreq[i]));
		}
	}


	if(num_gpus == 1){
    cudaSetDevice(selected);
		for(int i=0; i < data.total_frequencies; i++){
			gpuErrchk(cudaMalloc((void**)&device_vars[i].atten, sizeof(cufftComplex)*M*N));
			gpuErrchk(cudaMemset(device_vars[i].atten, 0, sizeof(cufftComplex)*M*N));
			gpuErrchk(cudaMalloc((void**)&device_vars[i].chi2, sizeof(float)*data.numVisibilitiesPerFreq[i]));
			gpuErrchk(cudaMemset(device_vars[i].chi2, 0, sizeof(float)*data.numVisibilitiesPerFreq[i]));


			gpuErrchk(cudaMalloc((void**)&device_vars[i].dchi2, sizeof(float)*M*N));
			gpuErrchk(cudaMemset(device_vars[i].dchi2, 0, sizeof(float)*M*N));



			gpuErrchk(cudaMemcpy(device_visibilities[i].u, visibilities[i].u, sizeof(float)*data.numVisibilitiesPerFreq[i], cudaMemcpyHostToDevice));
			gpuErrchk(cudaMemcpy(device_visibilities[i].v, visibilities[i].v, sizeof(float)*data.numVisibilitiesPerFreq[i], cudaMemcpyHostToDevice));
			gpuErrchk(cudaMemcpy(device_visibilities[i].weight, visibilities[i].weight, sizeof(float)*data.numVisibilitiesPerFreq[i], cudaMemcpyHostToDevice));
			gpuErrchk(cudaMemcpy(device_visibilities[i].Vo, visibilities[i].Vo, sizeof(cufftComplex)*data.numVisibilitiesPerFreq[i], cudaMemcpyHostToDevice));

			gpuErrchk(cudaMemset(device_visibilities[i].Vr, 0, sizeof(float)*data.numVisibilitiesPerFreq[i]));

		}
	}else{
		for(int i=0; i < data.total_frequencies; i++){
			cudaSetDevice(i%num_gpus);
			gpuErrchk(cudaMalloc((void**)&device_vars[i].atten, sizeof(cufftComplex)*M*N));
			gpuErrchk(cudaMemset(device_vars[i].atten, 0, sizeof(cufftComplex)*M*N));
			gpuErrchk(cudaMalloc((void**)&device_vars[i].chi2, sizeof(float)*data.numVisibilitiesPerFreq[i]));
			gpuErrchk(cudaMemset(device_vars[i].chi2, 0, sizeof(float)*data.numVisibilitiesPerFreq[i]));

			gpuErrchk(cudaMalloc((void**)&device_vars[i].dchi2, sizeof(float)*M*N));
			gpuErrchk(cudaMemset(device_vars[i].dchi2, 0, sizeof(float)*M*N));



			gpuErrchk(cudaMemcpy(device_visibilities[i].u, visibilities[i].u, sizeof(float)*data.numVisibilitiesPerFreq[i], cudaMemcpyHostToDevice));
			gpuErrchk(cudaMemcpy(device_visibilities[i].v, visibilities[i].v, sizeof(float)*data.numVisibilitiesPerFreq[i], cudaMemcpyHostToDevice));
			gpuErrchk(cudaMemcpy(device_visibilities[i].weight, visibilities[i].weight, sizeof(float)*data.numVisibilitiesPerFreq[i], cudaMemcpyHostToDevice));
			gpuErrchk(cudaMemcpy(device_visibilities[i].Vo, visibilities[i].Vo, sizeof(cufftComplex)*data.numVisibilitiesPerFreq[i], cudaMemcpyHostToDevice));
			gpuErrchk(cudaMemset(device_visibilities[i].Vr, 0, sizeof(float)*data.numVisibilitiesPerFreq[i]));
		}
	}

  //Declaring block size and number of blocks for Image
  dim3 threads(variables.blockSizeX, variables.blockSizeY);
	dim3 blocks(M/threads.x, N/threads.y);
	threadsPerBlockNN = threads;
	numBlocksNN = blocks;

	difmap_noise = beam_noise / (PI * beam_bmaj * beam_bmin / (4 * log(2) ));
  if(lambda == 0.0){
    MINPIX = 0.0;
  }else{
    MINPIX = 1.0 / minpix_factor;
  }

	float deltax = RPDEG*DELTAX; //radians
	float deltay = RPDEG*DELTAY; //radians
	deltau = 1.0 / (M * deltax);
	deltav = 1.0 / (N * deltay);



	cufftComplex *host_I = (cufftComplex*)malloc(M*N*sizeof(cufftComplex));
	cufftComplex *device_total_atten_image;
  /////////////////////////////////////////////////////CALCULATE DIRECTION COSINES/////////////////////////////////////////////////
	double lobs, mobs;
	double raimage = ra * RPDEG_D;
	double decimage = dec * RPDEG_D;
  if(verbose_flag){
  	printf("MS: Ra: %lf, dec: %lf\n", obsra, obsdec);
  	printf("FITS: Ra: %lf, dec: %lf\n", raimage, decimage);
  }

	direccos(obsra, obsdec, raimage, decimage, &lobs,  &mobs);

	global_xobs = (crpix1 - 1.0) + lobs/deltax;
	global_yobs = (crpix2 - 1.0) + mobs/deltay;
  if(verbose_flag){
	   printf("Image Center: %f, %f\n", global_xobs, global_yobs);
  }

	////////////////////////////////////////////////////////MAKE STARTING IMAGE////////////////////////////////////////////////////////


	for(int i=0;i<M;i++){
		for(int j=0;j<N;j++){
			host_I[N*i+j].x = MINPIX;
			host_I[N*i+j].y = 0;
		}
	}

	int it;
	////////////////////////////////////////////////CUDA MEMORY ALLOCATION FOR DEVICE///////////////////////////////////////////////////

	//exit(-1);


	if(num_gpus == 1){
    cudaSetDevice(selected);
		gpuErrchk(cudaMalloc((void**)&device_V, sizeof(cufftComplex)*M*N));
	  gpuErrchk(cudaMalloc((void**)&device_image, sizeof(cufftComplex)*M*N));
	}else{
		for (int i = 0;  i < data.total_frequencies; i++) {
			cudaSetDevice(i%num_gpus);
			gpuErrchk(cudaMalloc((void**)&device_vars[i].device_V, sizeof(cufftComplex)*M*N));
		  gpuErrchk(cudaMalloc((void**)&device_vars[i].device_image, sizeof(cufftComplex)*M*N));
		}
	}

  if(num_gpus == 1){
    cudaSetDevice(selected);
  }else{
	   cudaSetDevice(0);
  }
	gpuErrchk(cudaMalloc((void**)&device_I, sizeof(cufftComplex)*M*N));
  gpuErrchk(cudaMemset(device_I, 0, sizeof(cufftComplex)*M*N));

  gpuErrchk(cudaMemcpy2D(device_I, sizeof(cufftComplex), host_I, sizeof(cufftComplex), sizeof(cufftComplex), M*N, cudaMemcpyHostToDevice));

	gpuErrchk(cudaMalloc((void**)&device_total_atten_image, sizeof(cufftComplex)*M*N));
  gpuErrchk(cudaMemset(device_total_atten_image, 0, sizeof(cufftComplex)*M*N));

	gpuErrchk(cudaMalloc((void**)&device_noise_image, sizeof(cufftComplex)*M*N));
  gpuErrchk(cudaMemset(device_noise_image, 0, sizeof(cufftComplex)*M*N));

	gpuErrchk(cudaMalloc((void**)&device_fg_image, sizeof(cufftComplex)*M*N));
  gpuErrchk(cudaMemset(device_fg_image, 0, sizeof(cufftComplex)*M*N));

	gpuErrchk(cudaMalloc((void**)&device_dphi, sizeof(float)*M*N));
  gpuErrchk(cudaMemset(device_dphi, 0, sizeof(float)*M*N));

	gpuErrchk(cudaMalloc((void**)&device_dchi2_total, sizeof(float)*M*N));
  gpuErrchk(cudaMemset(device_dchi2_total, 0, sizeof(float)*M*N));


	gpuErrchk(cudaMalloc((void**)&device_dH, sizeof(float)*M*N));
  gpuErrchk(cudaMemset(device_dH, 0, sizeof(float)*M*N));

	gpuErrchk(cudaMalloc((void**)&device_H, sizeof(float)*M*N));
  gpuErrchk(cudaMemset(device_H, 0, sizeof(float)*M*N));



	if(num_gpus == 1){
    cudaSetDevice(selected);
		gpuErrchk(cudaMemset(device_V, 0, sizeof(cufftComplex)*M*N));
		gpuErrchk(cudaMemset(device_image, 0, sizeof(cufftComplex)*M*N));
	}else{
		for (int i = 0;  i < data.total_frequencies; i++) {
			cudaSetDevice(i%num_gpus);
			gpuErrchk(cudaMemset(device_vars[i].device_V, 0, sizeof(cufftComplex)*M*N));
			gpuErrchk(cudaMemset(device_vars[i].device_image, 0, sizeof(cufftComplex)*M*N));

		}
	}




	if(num_gpus == 1){
    cudaSetDevice(selected);
		if ((cufftPlan2d(&plan1GPU, N, M, CUFFT_C2C))!= CUFFT_SUCCESS) {
			printf("cufft plan error\n");
			return -1;
		}
	}else{
		for (int i = 0;  i < data.total_frequencies; i++) {
			cudaSetDevice(i%num_gpus);
			if ((cufftPlan2d(&device_vars[i].plan, N, M, CUFFT_C2C))!= CUFFT_SUCCESS) {
				printf("cufft plan error\n");
				return -1;
			}
		}

	}

  //Time is taken from first kernel
  t = clock();
  start = omp_get_wtime();
	if(num_gpus == 1){
    cudaSetDevice(selected);
		for(int i=0; i<data.total_frequencies; i++){
			hermitianSymmetry<<<visibilities[i].numBlocksUV, visibilities[i].threadsPerBlockUV>>>(device_visibilities[i].u, device_visibilities[i].v, device_visibilities[i].Vo, visibilities[i].freq, data.numVisibilitiesPerFreq[i]);
			gpuErrchk(cudaDeviceSynchronize());
		}
	}else{
		#pragma omp parallel for schedule(static,1)
    for (int i = 0; i < data.total_frequencies; i++)
		{
			unsigned int j = omp_get_thread_num();
			//unsigned int num_cpu_threads = omp_get_num_threads();
			// set and check the CUDA device for this CPU thread
			int gpu_id = -1;
			cudaSetDevice(i % num_gpus);   // "% num_gpus" allows more CPU threads than GPU devices
			cudaGetDevice(&gpu_id);
			hermitianSymmetry<<<visibilities[i].numBlocksUV, visibilities[i].threadsPerBlockUV>>>(device_visibilities[i].u, device_visibilities[i].v, device_visibilities[i].Vo, visibilities[i].freq, data.numVisibilitiesPerFreq[i]);
			gpuErrchk(cudaDeviceSynchronize());
		}

	}

	if(num_gpus == 1){
    cudaSetDevice(selected);
		for(int i=0; i<data.total_frequencies; i++){
			attenuation<<<numBlocksNN, threadsPerBlockNN>>>(device_vars[i].atten, visibilities[i].freq, N, global_xobs, global_yobs, DELTAX, DELTAY);
			gpuErrchk(cudaDeviceSynchronize());
      toFitsFloat(device_vars[i].atten, i, M, N, 4);
		}
	}else{
    #pragma omp parallel for schedule(static,1)
    for (int i = 0; i < data.total_frequencies; i++)
		{
      unsigned int j = omp_get_thread_num();
			//unsigned int num_cpu_threads = omp_get_num_threads();
			// set and check the CUDA device for this CPU thread
			int gpu_id = -1;
			cudaSetDevice(i % num_gpus);   // "% num_gpus" allows more CPU threads than GPU devices
			cudaGetDevice(&gpu_id);
			attenuation<<<numBlocksNN, threadsPerBlockNN>>>(device_vars[i].atten, visibilities[i].freq, N, global_xobs, global_yobs, DELTAX, DELTAY);
			gpuErrchk(cudaDeviceSynchronize());
		}

    for (int i = 0; i < data.total_frequencies; i++)
    {
      cudaSetDevice(i % num_gpus);   // "% num_gpus" allows more CPU threads than GPU devices
      toFitsFloat(device_vars[i].atten, i, M, N, 4);
    }
	}




	if(num_gpus == 1){
    cudaSetDevice(selected);
		for(int i=0; i<data.total_frequencies; i++){
			total_attenuation<<<numBlocksNN, threadsPerBlockNN>>>(device_total_atten_image, device_vars[i].atten, N);
			gpuErrchk(cudaDeviceSynchronize());

		}
	}else{
    #pragma omp parallel for schedule(static,1)
    for (int i = 0; i < data.total_frequencies; i++)
		{
      unsigned int j = omp_get_thread_num();
			//unsigned int num_cpu_threads = omp_get_num_threads();
			// set and check the CUDA device for this CPU thread
			int gpu_id = -1;
			cudaSetDevice(i % num_gpus);   // "% num_gpus" allows more CPU threads than GPU devices
			cudaGetDevice(&gpu_id);
			#pragma omp critical
			{
				total_attenuation<<<numBlocksNN, threadsPerBlockNN>>>(device_total_atten_image, device_vars[i].atten, N);
				gpuErrchk(cudaDeviceSynchronize());
			}
		}
	}


  if(num_gpus == 1){
    cudaSetDevice(selected);
		mean_attenuation<<<numBlocksNN, threadsPerBlockNN>>>(device_total_atten_image, data.total_frequencies, N);
		gpuErrchk(cudaDeviceSynchronize());
	}else{
    cudaSetDevice(0);
		mean_attenuation<<<numBlocksNN, threadsPerBlockNN>>>(device_total_atten_image, data.total_frequencies, N);
		gpuErrchk(cudaDeviceSynchronize());

	}

  toFitsFloat(device_total_atten_image, 0, M, N, 5);
  if(num_gpus == 1){
    cudaSetDevice(selected);
  }else{
	   cudaSetDevice(0);
   }

	noise_image<<<numBlocksNN, threadsPerBlockNN>>>(device_total_atten_image, device_noise_image, difmap_noise, N);
	gpuErrchk(cudaDeviceSynchronize());
	toFitsFloat(device_noise_image, 0, M, N, 6);
	//exit(0);
	cufftComplex *host_noise_image = (cufftComplex*)malloc(M*N*sizeof(cufftComplex));
	gpuErrchk(cudaMemcpy2D(host_noise_image, sizeof(cufftComplex), device_noise_image, sizeof(cufftComplex), sizeof(cufftComplex), M*N, cudaMemcpyDeviceToHost));
	for(int i=0; i<M; i++){
		for(int j=0; j<N; j++){
			if(host_noise_image[N*i+j].x < noise_min){
				noise_min = host_noise_image[N*i+j].x;
			}
		}
	}

	fg_scale = noise_min;
	noise_cut = noise_cut * noise_min;
  if(verbose_flag){
	   printf("fg_scale = %f\n", fg_scale);
  }
	free(host_noise_image);
  cudaFree(device_total_atten_image);
	//return;


	//////////////////////////////////////////////////////Fletcher-Reeves Polak-Ribiere Minimization////////////////////////////////////////////////////////////////
	printf("\n\nStarting Fletcher Reeves Polak Ribiere method (Conj. Grad.)\n\n");
	float fret = 0.0;
	frprmn(device_I	, ftol, &fret, chiCuadrado, dchiCuadrado);
  t = clock() - t;
  end = omp_get_wtime();
  printf("Minimization ended successfully\n");
	double time_taken = ((double)t)/CLOCKS_PER_SEC;
  double wall_time = end-start;
  printf("Total CPU time: %lf\n", time_taken);
  printf("Wall time: %lf\n\n\n", wall_time);

	//Pass residuals to host
	printf("Passing final image to disk\n");
	toFitsFloat(device_I, iter, M, N, 0);
  printf("Back UV coordinates to normal\n");
	if(num_gpus == 1){
    cudaSetDevice(selected);
		for(int i=0; i<data.total_frequencies; i++){
			backUV<<<visibilities[i].numBlocksUV, visibilities[i].threadsPerBlockUV>>>(device_visibilities[i].u, device_visibilities[i].v, visibilities[i].freq, data.numVisibilitiesPerFreq[i]);
			gpuErrchk(cudaDeviceSynchronize());
		}
	}else{
		#pragma omp parallel for schedule(static,1)
    for (int i = 0; i < data.total_frequencies; i++)
		{
			unsigned int j = omp_get_thread_num();
			//unsigned int num_cpu_threads = omp_get_num_threads();
			// set and check the CUDA device for this CPU thread
			int gpu_id = -1;
			cudaSetDevice(i % num_gpus);   // "% num_gpus" allows more CPU threads than GPU devices
			cudaGetDevice(&gpu_id);
			//printf("CPU thread %d takes frequency %d and uses CUDA device %d\n", j, i, gpu_id);
			backUV<<<visibilities[i].numBlocksUV, visibilities[i].threadsPerBlockUV>>>(device_visibilities[i].u, device_visibilities[i].v, visibilities[i].freq, data.numVisibilitiesPerFreq[i]);
			gpuErrchk(cudaDeviceSynchronize());
		}
	}


	//Saving residuals to disk
  residualsToHost(device_visibilities, visibilities, data);
  printf("Saving residuals to SQL...\n");
	writeMS(msoutput, visibilities);
	printf("Residuals saved.\n");

	//Free device and host memory
	printf("Free device and host memory\n");
	cufftDestroy(plan1GPU);
	for(int i=0; i<data.total_frequencies; i++){
		cudaSetDevice(i%num_gpus);
		cudaFree(device_visibilities[i].u);
		cudaFree(device_visibilities[i].v);
		cudaFree(device_visibilities[i].weight);

		cudaFree(device_visibilities[i].Vr);
		cudaFree(device_visibilities[i].Vo);
		cudaFree(device_vars[i].atten);

		cufftDestroy(device_vars[i].plan);
	}


	for(int i=0; i<data.total_frequencies; i++){
		free(visibilities[i].u);
		free(visibilities[i].v);
		free(visibilities[i].weight);
		free(visibilities[i].Vo);
	}

	cudaFree(device_I);
	if(num_gpus == 1){
		cudaFree(device_V);
		cudaFree(device_image);
	}else{
		for(int i=0; i<data.total_frequencies; i++){
			cudaSetDevice(i%num_gpus);
			cudaFree(device_vars[i].device_V);
			cudaFree(device_vars[i].device_image);
		}
	}
  if(num_gpus == 1){
    cudaSetDevice(selected);
  }else{
    cudaSetDevice(0);
  }
	cudaFree(device_total_atten_image);
	cudaFree(device_noise_image);
	cudaFree(device_fg_image);

	cudaFree(device_dphi);
	cudaFree(device_dchi2_total);
	cudaFree(device_dH);

	cudaFree(device_chi2);
	cudaFree(device_H);

  //Disabling UVA
  if(num_gpus > 1){
    for(int i=1; i<num_gpus; i++){
          cudaSetDevice(0);
          cudaDeviceDisablePeerAccess(i);
          cudaSetDevice(i%num_gpus);
          cudaDeviceDisablePeerAccess(0);
    }

    for(int i=0; i<num_gpus; i++ ){
          cudaSetDevice(i%num_gpus);
          cudaDeviceReset();
    }
  }
	free(host_I);
	free(msinput);
	free(msoutput);
	free(modinput);
	free(beaminput);

  fits_close_file(mod_in, &status_mod_in);
  if (status_mod_in) {
    fits_report_error(stderr, status_mod_in); /* print error message */
    goToError();
  }

	return 0;
}
