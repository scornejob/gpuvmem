#include "frprmn.cuh"
#include "directioncosines.cuh"
#include <time.h>

long M, N, numVisibilities;
int iter=0;

cufftHandle plan1GPU;

cufftComplex *device_I, *device_V, *device_fg_image, *device_image, *device_noise_image, *device_weight_image;

float *device_dphi, *device_dchi2_total, *device_dH, *device_chi2, *device_H, DELTAX, DELTAY, deltau, deltav, beam_noise, beam_bmaj;
float beam_bmin, b_noise_aux, noise_cut, MINPIX, minpix_factor, lambda, ftol, random_probability;
float difmap_noise, fg_scale, final_chi2, final_H, beam_fwhm, beam_freq, beam_cutoff;

dim3 threadsPerBlockNN;
dim3 numBlocksNN;

int threadsVectorReduceNN, blocksVectorReduceNN, crpix1, crpix2, nopositivity = 0, nsamples, nfields, nstokes, verbose_flag = 0, xcorr_flag = 0, clip_flag = 0, it_maximum, status_mod_in;
int num_gpus, multigpu, firstgpu, selected, t_telescope;
char *output, *mempath, *out_image;

double ra, dec;

freqData data;
fitsfile *mod_in;

Field *fields;
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


	float noise_min = 1E32;

	Vars variables = getOptions(argc, argv);
	char *msinput = variables.input;
	char *msoutput = variables.output;
  char *inputdat = variables.inputdat;
	char *modinput = variables.modin;
  out_image = variables.output_image;
  selected = variables.select;
  mempath = variables.path;
  it_maximum = variables.it_max;
  int total_visibilities = 0;
  b_noise_aux = variables.noise;
  lambda = variables.lambda;
  multigpu = 0;
  firstgpu = -1;

  struct stat st = {0};

  if(stat(mempath, &st) == -1) mkdir(mempath,0700);

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

  if(selected > num_gpus || selected < 0){
    printf("ERROR. THE SELECTED GPU DOESN'T EXIST\n");
    exit(-1);
  }

  readInputDat(inputdat);
  init_beam(t_telescope);
	data = getFreqs(msinput);
  if(verbose_flag){
	   printf("Number of frequencies file = %d\n", data.total_frequencies);
  }

  if(strcmp(variables.multigpu, "NULL")!=0){
    //Counts number of gpus to use
    char *pt;
    pt = strtok(variables.multigpu,",");

    while(pt!=NULL){
      if(multigpu==0){
        firstgpu = atoi(pt);
      }
      multigpu++;
      pt = strtok (NULL, ",");
    }
  }else{
    multigpu = 0;
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
	  for(int i=firstgpu + 1; i< firstgpu + num_gpus; i++){
			cudaDeviceProp dprop0, dpropX;
			cudaGetDeviceProperties(&dprop0, firstgpu);
			cudaGetDeviceProperties(&dpropX, i);
			int canAccessPeer0_x, canAccessPeerx_0;
			cudaDeviceCanAccessPeer(&canAccessPeer0_x, firstgpu, i);
			cudaDeviceCanAccessPeer(&canAccessPeerx_0 , i, firstgpu);
      if(verbose_flag){
  			printf("> Peer-to-Peer (P2P) access from %s (GPU%d) -> %s (GPU%d) : %s\n", dprop0.name, firstgpu, dpropX.name, i, canAccessPeer0_x ? "Yes" : "No");
      	printf("> Peer-to-Peer (P2P) access from %s (GPU%d) -> %s (GPU%d) : %s\n", dpropX.name, i, dprop0.name, firstgpu, canAccessPeerx_0 ? "Yes" : "No");
      }
			if(canAccessPeer0_x == 0 || canAccessPeerx_0 == 0){
				printf("Two or more SM 2.0 class GPUs are required for %s to run.\n", argv[0]);
        printf("Support for UVA requires a GPU with SM 2.0 capabilities.\n");
        printf("Peer to Peer access is not available between GPU%d <-> GPU%d, waiving test.\n", 0, i);
        exit(EXIT_SUCCESS);
			}else{
				cudaSetDevice(firstgpu);
        if(verbose_flag){
          printf("Granting access from %d to %d...\n",firstgpu, i);
        }
				cudaDeviceEnablePeerAccess(i,firstgpu);
				cudaSetDevice((i%num_gpus) + firstgpu);
        if(verbose_flag){
          printf("Granting access from %d to %d...\n", i, firstgpu);
        }
				cudaDeviceEnablePeerAccess(firstgpu,firstgpu);
        if(verbose_flag){
				      printf("Checking GPU %d and GPU %d for UVA capabilities...\n", firstgpu, i);
        }
				const bool has_uva = (dprop0.unifiedAddressing && dpropX.unifiedAddressing);
        if(verbose_flag){
  				printf("> %s (GPU%d) supports UVA: %s\n", dprop0.name, firstgpu, (dprop0.unifiedAddressing ? "Yes" : "No"));
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

  for(int f=0; f<nfields; f++){
  	fields[f].visibilities = (Vis*)malloc(data.total_frequencies*sizeof(Vis));
  	fields[f].device_visibilities = (Vis*)malloc(data.total_frequencies*sizeof(Vis));
  	fields[f].device_vars = (VPF*)malloc(data.total_frequencies*sizeof(VPF));
  }

  //ALLOCATE MEMORY AND GET TOTAL NUMBER OF VISIBILITIES
  for(int f=0; f<nfields; f++){
  	for(int i=0; i < data.total_frequencies; i++){
  		fields[f].visibilities[i].stokes = (int*)malloc(fields[f].numVisibilitiesPerFreq[i]*sizeof(int));
  		fields[f].visibilities[i].u = (float*)malloc(fields[f].numVisibilitiesPerFreq[i]*sizeof(float));
  		fields[f].visibilities[i].v = (float*)malloc(fields[f].numVisibilitiesPerFreq[i]*sizeof(float));
  		fields[f].visibilities[i].weight = (float*)malloc(fields[f].numVisibilitiesPerFreq[i]*sizeof(float));
  		fields[f].visibilities[i].Vo = (cufftComplex*)malloc(fields[f].numVisibilitiesPerFreq[i]*sizeof(cufftComplex));
      fields[f].visibilities[i].Vm = (cufftComplex*)malloc(fields[f].numVisibilitiesPerFreq[i]*sizeof(cufftComplex));
      total_visibilities += fields[f].numVisibilitiesPerFreq[i];
  	}
  }


  if(verbose_flag){
	   printf("Reading visibilities and FITS input files...\n");
  }
	readMS(msinput, modinput, fields);

  if(verbose_flag){
    printf("MS File Successfully Read\n");
    if(beam_noise == -1){
      printf("Beam noise wasn't provided by the user... Calculating...\n");
    }
  }

  //Declaring block size and number of blocks for visibilities
  float sum_inverse_weight = 0.0;
  for(int f=0; f<nfields; f++){
  	for(int i=0; i< data.total_frequencies; i++){
      if(beam_noise == -1){
        //Calculating beam noise
        for(int j=0; j< fields[f].numVisibilitiesPerFreq[i]; j++){
            sum_inverse_weight += 1/fields[f].visibilities[i].weight[j];
        }
      }
  		fields[f].visibilities[i].numVisibilities = fields[f].numVisibilitiesPerFreq[i];
  		long UVpow2 = NearestPowerOf2(fields[f].visibilities[i].numVisibilities);
      fields[f].visibilities[i].threadsPerBlockUV = variables.blockSizeV;
  		fields[f].visibilities[i].numBlocksUV = UVpow2/fields[f].visibilities[i].threadsPerBlockUV;
    }
  }

  if(beam_noise == -1){
      beam_noise = sqrt(sum_inverse_weight)/total_visibilities;
      printf("Noise: %e\n", beam_noise);
  }

	if(num_gpus == 1){
    cudaSetDevice(selected);
    for(int f=0; f<nfields; f++){
  		for(int i=0; i<data.total_frequencies; i++){
  			gpuErrchk(cudaMalloc(&fields[f].device_visibilities[i].u, sizeof(float)*fields[f].numVisibilitiesPerFreq[i]));
  			gpuErrchk(cudaMalloc(&fields[f].device_visibilities[i].v, sizeof(float)*fields[f].numVisibilitiesPerFreq[i]));
  			gpuErrchk(cudaMalloc(&fields[f].device_visibilities[i].Vo, sizeof(cufftComplex)*fields[f].numVisibilitiesPerFreq[i]));
  			gpuErrchk(cudaMalloc(&fields[f].device_visibilities[i].weight, sizeof(float)*fields[f].numVisibilitiesPerFreq[i]));
        gpuErrchk(cudaMalloc(&fields[f].device_visibilities[i].Vm, sizeof(cufftComplex)*fields[f].numVisibilitiesPerFreq[i]));
  			gpuErrchk(cudaMalloc(&fields[f].device_visibilities[i].Vr, sizeof(cufftComplex)*fields[f].numVisibilitiesPerFreq[i]));
  		}
    }
	}else{
    for(int f=0; f<nfields; f++){
  		for(int i=0; i<data.total_frequencies; i++){
  			cudaSetDevice((i%num_gpus) + firstgpu);
  			gpuErrchk(cudaMalloc(&fields[f].device_visibilities[i].u, sizeof(float)*fields[f].numVisibilitiesPerFreq[i]));
  			gpuErrchk(cudaMalloc(&fields[f].device_visibilities[i].v, sizeof(float)*fields[f].numVisibilitiesPerFreq[i]));
  			gpuErrchk(cudaMalloc(&fields[f].device_visibilities[i].Vo, sizeof(cufftComplex)*fields[f].numVisibilitiesPerFreq[i]));
  			gpuErrchk(cudaMalloc(&fields[f].device_visibilities[i].weight, sizeof(float)*fields[f].numVisibilitiesPerFreq[i]));
        gpuErrchk(cudaMalloc(&fields[f].device_visibilities[i].Vm, sizeof(cufftComplex)*fields[f].numVisibilitiesPerFreq[i]));
  			gpuErrchk(cudaMalloc(&fields[f].device_visibilities[i].Vr, sizeof(cufftComplex)*fields[f].numVisibilitiesPerFreq[i]));
  		}
    }
	}


	if(num_gpus == 1){
    cudaSetDevice(selected);
    for(int f=0; f<nfields; f++){
      gpuErrchk(cudaMalloc((void**)&fields[f].atten_image, sizeof(cufftComplex)*M*N));
      gpuErrchk(cudaMemset(fields[f].atten_image, 0, sizeof(cufftComplex)*M*N));
  		for(int i=0; i < data.total_frequencies; i++){
  			gpuErrchk(cudaMalloc((void**)&fields[f].device_vars[i].atten, sizeof(cufftComplex)*M*N));
  			gpuErrchk(cudaMemset(fields[f].device_vars[i].atten, 0, sizeof(cufftComplex)*M*N));

  			gpuErrchk(cudaMalloc(&fields[f].device_vars[i].chi2, sizeof(float)*fields[f].numVisibilitiesPerFreq[i]));
  			gpuErrchk(cudaMemset(fields[f].device_vars[i].chi2, 0, sizeof(float)*fields[f].numVisibilitiesPerFreq[i]));

        if(xcorr_flag){
          gpuErrchk(cudaMalloc(&fields[f].device_vars[i].alpha_num, sizeof(float)*fields[f].numVisibilitiesPerFreq[i]));
    			gpuErrchk(cudaMemset(fields[f].device_vars[i].alpha_num, 0, sizeof(float)*fields[f].numVisibilitiesPerFreq[i]));
          gpuErrchk(cudaMalloc(&fields[f].device_vars[i].alpha_den, sizeof(float)*fields[f].numVisibilitiesPerFreq[i]));
    			gpuErrchk(cudaMemset(fields[f].device_vars[i].alpha_den, 0, sizeof(float)*fields[f].numVisibilitiesPerFreq[i]));
        }


  			gpuErrchk(cudaMalloc((void**)&fields[f].device_vars[i].dchi2, sizeof(float)*M*N));
  			gpuErrchk(cudaMemset(fields[f].device_vars[i].dchi2, 0, sizeof(float)*M*N));

  			gpuErrchk(cudaMemcpy(fields[f].device_visibilities[i].u, fields[f].visibilities[i].u, sizeof(float)*fields[f].numVisibilitiesPerFreq[i], cudaMemcpyHostToDevice));

  			gpuErrchk(cudaMemcpy(fields[f].device_visibilities[i].v, fields[f].visibilities[i].v, sizeof(float)*fields[f].numVisibilitiesPerFreq[i], cudaMemcpyHostToDevice));

  			gpuErrchk(cudaMemcpy(fields[f].device_visibilities[i].weight, fields[f].visibilities[i].weight, sizeof(float)*fields[f].numVisibilitiesPerFreq[i], cudaMemcpyHostToDevice));

  			gpuErrchk(cudaMemcpy(fields[f].device_visibilities[i].Vo, fields[f].visibilities[i].Vo, sizeof(cufftComplex)*fields[f].numVisibilitiesPerFreq[i], cudaMemcpyHostToDevice));

  			gpuErrchk(cudaMemset(fields[f].device_visibilities[i].Vr, 0, sizeof(cufftComplex)*fields[f].numVisibilitiesPerFreq[i]));
        gpuErrchk(cudaMemset(fields[f].device_visibilities[i].Vm, 0, sizeof(cufftComplex)*fields[f].numVisibilitiesPerFreq[i]));

  		}
    }
	}else{
    for(int f=0; f<nfields; f++){
      cudaSetDevice(0);
      gpuErrchk(cudaMalloc((void**)&fields[f].atten_image, sizeof(cufftComplex)*M*N));
      gpuErrchk(cudaMemset(fields[f].atten_image, 0, sizeof(cufftComplex)*M*N));
  		for(int i=0; i < data.total_frequencies; i++){
  			cudaSetDevice((i%num_gpus) + firstgpu);
  			gpuErrchk(cudaMalloc((void**)&fields[f].device_vars[i].atten, sizeof(cufftComplex)*M*N));
  			gpuErrchk(cudaMemset(fields[f].device_vars[i].atten, 0, sizeof(cufftComplex)*M*N));
  			gpuErrchk(cudaMalloc(&fields[f].device_vars[i].chi2, sizeof(float)*fields[f].numVisibilitiesPerFreq[i]));
  			gpuErrchk(cudaMemset(fields[f].device_vars[i].chi2, 0, sizeof(float)*fields[f].numVisibilitiesPerFreq[i]));

        if(xcorr_flag){
          gpuErrchk(cudaMalloc(&fields[f].device_vars[i].alpha_num, sizeof(float)*fields[f].numVisibilitiesPerFreq[i]));
    			gpuErrchk(cudaMemset(fields[f].device_vars[i].alpha_num, 0, sizeof(float)*fields[f].numVisibilitiesPerFreq[i]));
          gpuErrchk(cudaMalloc(&fields[f].device_vars[i].alpha_den, sizeof(float)*fields[f].numVisibilitiesPerFreq[i]));
    			gpuErrchk(cudaMemset(fields[f].device_vars[i].alpha_den, 0, sizeof(float)*fields[f].numVisibilitiesPerFreq[i]));
        }

  			gpuErrchk(cudaMalloc((void**)&fields[f].device_vars[i].dchi2, sizeof(float)*M*N));
  			gpuErrchk(cudaMemset(fields[f].device_vars[i].dchi2, 0, sizeof(float)*M*N));

  			gpuErrchk(cudaMemcpy(fields[f].device_visibilities[i].u, fields[f].visibilities[i].u, sizeof(float)*fields[f].numVisibilitiesPerFreq[i], cudaMemcpyHostToDevice));

  			gpuErrchk(cudaMemcpy(fields[f].device_visibilities[i].v, fields[f].visibilities[i].v, sizeof(float)*fields[f].numVisibilitiesPerFreq[i], cudaMemcpyHostToDevice));

  			gpuErrchk(cudaMemcpy(fields[f].device_visibilities[i].weight, fields[f].visibilities[i].weight, sizeof(float)*fields[f].numVisibilitiesPerFreq[i], cudaMemcpyHostToDevice));

  			gpuErrchk(cudaMemcpy(fields[f].device_visibilities[i].Vo, fields[f].visibilities[i].Vo, sizeof(cufftComplex)*fields[f].numVisibilitiesPerFreq[i], cudaMemcpyHostToDevice));

  			gpuErrchk(cudaMemset(fields[f].device_visibilities[i].Vr, 0, sizeof(float)*fields[f].numVisibilitiesPerFreq[i]));
        gpuErrchk(cudaMemset(fields[f].device_visibilities[i].Vm, 0, sizeof(cufftComplex)*fields[f].numVisibilitiesPerFreq[i]));
  		}
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
  /////////////////////////////////////////////////////CALCULATE DIRECTION COSINES/////////////////////////////////////////////////
  double raimage = ra * RPDEG_D;
  double decimage = dec * RPDEG_D;
  if(verbose_flag){
    printf("FITS: Ra: %lf, dec: %lf\n", raimage, decimage);
  }
  for(int f=0; f<nfields; f++){
  	double lobs, mobs;

  	direccos(fields[f].obsra, fields[f].obsdec, raimage, decimage, &lobs,  &mobs);

  	fields[f].global_xobs = (crpix1 - 1.0) - lobs/deltax;
  	fields[f].global_yobs = (crpix2 - 1.0) - mobs/deltay;
    if(verbose_flag){
  	   printf("Field %d - Ra: %f, dec: %f , x0: %f, y0: %f\n", f, fields[f].obsra, fields[f].obsdec, fields[f].global_xobs, fields[f].global_yobs);
    }

    if(fields[f].global_xobs < 0 || fields[f].global_xobs > M || fields[f].global_xobs < 0 || fields[f].global_yobs > N) {
      printf("Pointing center (%f,%f) is outside the range of the image\n", fields[f].global_xobs, fields[f].global_xobs);
      goToError();
    }
  }
	////////////////////////////////////////////////////////MAKE STARTING IMAGE////////////////////////////////////////////////////////


	for(int i=0;i<M;i++){
		for(int j=0;j<N;j++){
			host_I[N*i+j].x = MINPIX;
			host_I[N*i+j].y = 0;
		}
	}
	////////////////////////////////////////////////CUDA MEMORY ALLOCATION FOR DEVICE///////////////////////////////////////////////////

	if(num_gpus == 1){
    cudaSetDevice(selected);
		gpuErrchk(cudaMalloc((void**)&device_V, sizeof(cufftComplex)*M*N));
	  gpuErrchk(cudaMalloc((void**)&device_image, sizeof(cufftComplex)*M*N));
	}else{
    for(int f = 0; f<nfields; f++){
  		for (int i = 0;  i < data.total_frequencies; i++) {
  			cudaSetDevice((i%num_gpus) + firstgpu);
  			gpuErrchk(cudaMalloc((void**)&fields[f].device_vars[i].device_V, sizeof(cufftComplex)*M*N));
  		  gpuErrchk(cudaMalloc((void**)&fields[f].device_vars[i].device_image, sizeof(cufftComplex)*M*N));
  		}
    }
	}

  if(num_gpus == 1){
    cudaSetDevice(selected);
  }else{
	   cudaSetDevice(firstgpu);
  }
	gpuErrchk(cudaMalloc((void**)&device_I, sizeof(cufftComplex)*M*N));
  gpuErrchk(cudaMemset(device_I, 0, sizeof(cufftComplex)*M*N));

  gpuErrchk(cudaMemcpy2D(device_I, sizeof(cufftComplex), host_I, sizeof(cufftComplex), sizeof(cufftComplex), M*N, cudaMemcpyHostToDevice));

  gpuErrchk(cudaMalloc((void**)&device_noise_image, sizeof(cufftComplex)*M*N));
  gpuErrchk(cudaMemset(device_noise_image, 0, sizeof(cufftComplex)*M*N));

  gpuErrchk(cudaMalloc((void**)&device_weight_image, sizeof(cufftComplex)*M*N));
  gpuErrchk(cudaMemset(device_weight_image, 0, sizeof(cufftComplex)*M*N));

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
    for(int f = 0; f < nfields; f++){
  		for (int i = 0;  i < data.total_frequencies; i++) {
  			cudaSetDevice((i%num_gpus) + firstgpu);
  			gpuErrchk(cudaMemset(fields[f].device_vars[i].device_V, 0, sizeof(cufftComplex)*M*N));
  			gpuErrchk(cudaMemset(fields[f].device_vars[i].device_image, 0, sizeof(cufftComplex)*M*N));

  		}
    }
	}




	if(num_gpus == 1){
    cudaSetDevice(selected);
		if ((cufftPlan2d(&plan1GPU, N, M, CUFFT_C2C))!= CUFFT_SUCCESS) {
			printf("cufft plan error\n");
			return -1;
		}
	}else{
    for(int f = 0; f < nfields; f++){
  		for (int i = 0;  i < data.total_frequencies; i++) {
  			cudaSetDevice((i%num_gpus) + firstgpu);
  			if ((cufftPlan2d(&fields[f].device_vars[i].plan, N, M, CUFFT_C2C))!= CUFFT_SUCCESS) {
  				printf("cufft plan error\n");
  				return -1;
  			}
  		}
    }
	}

  //Time is taken from first kernel
  t = clock();
  start = omp_get_wtime();
	if(num_gpus == 1){
    cudaSetDevice(selected);
    for(int f=0; f < nfields; f++){
  		for(int i=0; i<data.total_frequencies; i++){
  			hermitianSymmetry<<<fields[f].visibilities[i].numBlocksUV, fields[f].visibilities[i].threadsPerBlockUV>>>(fields[f].device_visibilities[i].u, fields[f].device_visibilities[i].v, fields[f].device_visibilities[i].Vo, fields[f].visibilities[i].freq, fields[f].numVisibilitiesPerFreq[i]);
  			gpuErrchk(cudaDeviceSynchronize());
  		}
    }
	}else{
    for(int f = 0; f < nfields; f++){
  		#pragma omp parallel for schedule(static,1)
      for (int i = 0; i < data.total_frequencies; i++)
  		{
  			unsigned int j = omp_get_thread_num();
  			//unsigned int num_cpu_threads = omp_get_num_threads();
  			// set and check the CUDA device for this CPU thread
  			int gpu_id = -1;
  			cudaSetDevice((i%num_gpus) + firstgpu);   // "% num_gpus" allows more CPU threads than GPU devices
  			cudaGetDevice(&gpu_id);
  			hermitianSymmetry<<<fields[f].visibilities[i].numBlocksUV, fields[f].visibilities[i].threadsPerBlockUV>>>(fields[f].device_visibilities[i].u, fields[f].device_visibilities[i].v, fields[f].device_visibilities[i].Vo, fields[f].visibilities[i].freq, fields[f].numVisibilitiesPerFreq[i]);
  			gpuErrchk(cudaDeviceSynchronize());
  		}

  	}
  }

	if(num_gpus == 1){
    cudaSetDevice(selected);
    for(int f = 0; f<nfields; f++){
  		for(int i=0; i<data.total_frequencies; i++){
        if(fields[f].numVisibilitiesPerFreq[i] > 0){
    			attenuation<<<numBlocksNN, threadsPerBlockNN>>>(beam_fwhm, beam_freq, beam_cutoff, fields[f].device_vars[i].atten, fields[f].visibilities[i].freq, N, fields[f].global_xobs, fields[f].global_yobs, DELTAX, DELTAY);
    			gpuErrchk(cudaDeviceSynchronize());
        }
  		}
    }
	}else{
    for(int f = 0; f<nfields; f++){
      #pragma omp parallel for schedule(static,1)
      for (int i = 0; i < data.total_frequencies; i++)
  		{
        unsigned int j = omp_get_thread_num();
  			//unsigned int num_cpu_threads = omp_get_num_threads();
  			// set and check the CUDA device for this CPU thread
  			int gpu_id = -1;
  			cudaSetDevice((i%num_gpus) + firstgpu);   // "% num_gpus" allows more CPU threads than GPU devices
  			cudaGetDevice(&gpu_id);
        if(fields[f].numVisibilitiesPerFreq[i] > 0){
    			attenuation<<<numBlocksNN, threadsPerBlockNN>>>(beam_fwhm, beam_freq, beam_cutoff, fields[f].device_vars[i].atten, fields[f].visibilities[i].freq, N, fields[f].global_xobs, fields[f].global_yobs, DELTAX, DELTAY);
    			gpuErrchk(cudaDeviceSynchronize());
        }
  		}
    }
	}


	if(num_gpus == 1){
    cudaSetDevice(selected);
    for(int f=0; f<nfields; f++){
  		for(int i=0; i<data.total_frequencies; i++){
        if(fields[f].numVisibilitiesPerFreq[i] > 0){
    			total_attenuation<<<numBlocksNN, threadsPerBlockNN>>>(fields[f].atten_image, fields[f].device_vars[i].atten, N);
    			gpuErrchk(cudaDeviceSynchronize());
        }
  		}
    }
	}else{
    for(int f=0; f<nfields; f++){
      #pragma omp parallel for schedule(static,1)
      for (int i = 0; i < data.total_frequencies; i++)
  		{
        unsigned int j = omp_get_thread_num();
  			//unsigned int num_cpu_threads = omp_get_num_threads();
  			// set and check the CUDA device for this CPU thread
  			int gpu_id = -1;
  			cudaSetDevice((i%num_gpus) + firstgpu);   // "% num_gpus" allows more CPU threads than GPU devices
  			cudaGetDevice(&gpu_id);
        if(fields[f].numVisibilitiesPerFreq[i] > 0){
    			#pragma omp critical
    			{
    				total_attenuation<<<numBlocksNN, threadsPerBlockNN>>>(fields[f].atten_image, fields[f].device_vars[i].atten, N);
    				gpuErrchk(cudaDeviceSynchronize());
    			}
        }
  		}
  	}
  }

  for(int f=0; f<nfields; f++){
    if(num_gpus == 1){
      cudaSetDevice(selected);
    	mean_attenuation<<<numBlocksNN, threadsPerBlockNN>>>(fields[f].atten_image, fields[f].valid_frequencies, N);
    	gpuErrchk(cudaDeviceSynchronize());
  	}else{
      cudaSetDevice(firstgpu);
    	mean_attenuation<<<numBlocksNN, threadsPerBlockNN>>>(fields[f].atten_image, fields[f].valid_frequencies, N);
    	gpuErrchk(cudaDeviceSynchronize());
  	}
    toFitsFloat(fields[f].atten_image, f, M, N, 4);
  }

  if(num_gpus == 1){
    cudaSetDevice(selected);
  }else{
	   cudaSetDevice(firstgpu);
  }

  for(int f=0; f<nfields; f++){
    weight_image<<<numBlocksNN, threadsPerBlockNN>>>(device_weight_image, fields[f].atten_image, difmap_noise, N);
    gpuErrchk(cudaDeviceSynchronize());
  }
  noise_image<<<numBlocksNN, threadsPerBlockNN>>>(device_noise_image, device_weight_image, difmap_noise, N);
  gpuErrchk(cudaDeviceSynchronize());
  toFitsFloat(device_noise_image, 0, M, N, 5);


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
	   printf("fg_scale = %e\n", fg_scale);
     printf("difmap_noise = %e\n", difmap_noise);
  }
	free(host_noise_image);
  cudaFree(device_weight_image);
  for(int f=0; f<nfields; f++){
    cudaFree(fields[f].atten_image);
  }



	//////////////////////////////////////////////////////Fletcher-Reeves Polak-Ribiere Minimization////////////////////////////////////////////////////////////////
	printf("\n\nStarting Fletcher Reeves Polak Ribiere method (Conj. Grad.)\n\n");
	float fret = 0.0;
	frprmn(device_I	, ftol, &fret, chiCuadrado, dchiCuadrado);
  t = clock() - t;
  end = omp_get_wtime();
  printf("Minimization ended successfully\n\n");
  printf("Iterations: %d\n", iter);
  printf("chi2: %f\n", final_chi2);
  printf("0.5*chi2: %f\n", 0.5*final_chi2);
  printf("Total visibilities: %d\n", total_visibilities);
  printf("Reduced-chi2: %f\n", (0.5*final_chi2)/total_visibilities);
  printf("S: %f\n", final_H);
  printf("lambda*S: %f\n\n", lambda*final_H);
	double time_taken = ((double)t)/CLOCKS_PER_SEC;
  double wall_time = end-start;
  printf("Total CPU time: %lf\n", time_taken);
  printf("Wall time: %lf\n\n\n", wall_time);
  chiCuadrado(device_I);
	//Pass residuals to host
	printf("Saving final image to disk\n");
	toFitsFloat(device_I, iter, M, N, 0);
	//Saving residuals to disk
  residualsToHost(fields, data);
  printf("Saving residuals to MS...\n");
	writeMS(msinput,msoutput,fields);
	printf("Residuals saved.\n");

	//Free device and host memory
	printf("Free device and host memory\n");
	cufftDestroy(plan1GPU);
  for(int f=0; f<nfields; f++){
  	for(int i=0; i<data.total_frequencies; i++){
      if(num_gpus > 1){
  		    cudaSetDevice((i%num_gpus) + firstgpu);
      }
  		cudaFree(fields[f].device_visibilities[i].u);
  		cudaFree(fields[f].device_visibilities[i].v);
  		cudaFree(fields[f].device_visibilities[i].weight);

  		cudaFree(fields[f].device_visibilities[i].Vr);
  		cudaFree(fields[f].device_visibilities[i].Vo);
  		cudaFree(fields[f].device_vars[i].atten);

      if(xcorr_flag){
        cudaFree(fields[f].device_vars[i].alpha_num);
        cudaFree(fields[f].device_vars[i].alpha_den);
      }

  		cufftDestroy(fields[f].device_vars[i].plan);
  	}
  }

  for(int f=0; f<nfields; f++){
  	for(int i=0; i<data.total_frequencies; i++){
  		free(fields[f].visibilities[i].u);
  		free(fields[f].visibilities[i].v);
  		free(fields[f].visibilities[i].weight);
  		free(fields[f].visibilities[i].Vo);
  	}
  }

	cudaFree(device_I);
	if(num_gpus == 1){
		cudaFree(device_V);
		cudaFree(device_image);
	}else{
    for(int f=0; f<nfields;f++){
  		for(int i=0; i<data.total_frequencies; i++){
  			cudaSetDevice((i%num_gpus) + firstgpu);
  			cudaFree(fields[f].device_vars[i].device_V);
  			cudaFree(fields[f].device_vars[i].device_image);
  		}
    }
	}
  if(num_gpus == 1){
    cudaSetDevice(selected);
  }else{
    cudaSetDevice(firstgpu);
  }

	cudaFree(device_noise_image);
	cudaFree(device_fg_image);

	cudaFree(device_dphi);
	cudaFree(device_dchi2_total);
	cudaFree(device_dH);

	cudaFree(device_chi2);
	cudaFree(device_H);

  //Disabling UVA
  if(num_gpus > 1){
    for(int i=firstgpu+1; i<num_gpus+firstgpu; i++){
          cudaSetDevice(firstgpu);
          cudaDeviceDisablePeerAccess(i);
          cudaSetDevice((i%num_gpus) + firstgpu);
          cudaDeviceDisablePeerAccess(firstgpu);
    }

    for(int i=0; i<num_gpus; i++ ){
          cudaSetDevice((i%num_gpus) + firstgpu);
          cudaDeviceReset();
    }
  }
	free(host_I);
	free(msinput);
	free(msoutput);
	free(modinput);

  fits_close_file(mod_in, &status_mod_in);
  if (status_mod_in) {
    fits_report_error(stderr, status_mod_in);
    goToError();
  }

	return 0;
}
