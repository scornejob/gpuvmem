#include "synthesizer.cuh"
#include "imageProcessor.cuh"


long M, N, numVisibilities;
int iter=0;

cufftHandle plan1GPU;

cufftComplex *device_V, *device_fg_image, *device_image;

float *device_Image, *device_dphi, *device_chi2, *device_dchi2_total, *device_dS, *device_dchi2, *device_S, DELTAX, DELTAY, deltau, deltav, beam_noise, beam_bmaj, *device_noise_image, *device_weight_image;
float beam_bmin, b_noise_aux, noise_cut, MINPIX, minpix, lambda, ftol, random_probability = 1.0;
float noise_jypix, fg_scale, final_chi2, final_S, antenna_diameter, pb_factor, pb_cutoff, eta, robust_param;
float *host_I, sum_weights, *initial_values, *penalizators;
Telescope *telescope;

dim3 threadsPerBlockNN;
dim3 numBlocksNN;

int threadsVectorReduceNN, blocksVectorReduceNN, crpix1, crpix2, nopositivity = 0, verbose_flag = 0, clip_flag = 0, apply_noise = 0, print_images = 0, gridding, it_maximum, status_mod_in;
int multigpu, firstgpu, selected, t_telescope, reg_term, total_visibilities, image_count, nPenalizators, print_errors;
char *output, *mempath, *out_image, *msinput, *msoutput, *inputdat, *modinput;
float nu_0, threshold;

extern int num_gpus;

double ra, dec;

freqData data;

fitsfile *mod_in;

Field *fields;

VariablesPerField *vars_per_field;

varsPerGPU *vars_gpu;

Vars variables;

clock_t t;
double start, end;

float noise_min = 1E32;

inline bool IsGPUCapableP2P(cudaDeviceProp *pProp)
{
  #ifdef _WIN32
        return (bool)(pProp->tccDriver ? true : false);
  #else
        return (bool)(pProp->major >= 2);
  #endif
}

void MFS::configure(int argc, char **argv)
{
        if(iohandler == NULL)
        {
                iohandler = Singleton<IoFactory>::Instance().CreateIo(0);
        }

        variables = getOptions(argc, argv);
        msinput = variables.input;
        msoutput = variables.output;
        inputdat = variables.inputdat;
        modinput = variables.modin;
        out_image = variables.output_image;
        selected = variables.select;
        mempath = variables.path;
        it_maximum = variables.it_max;
        total_visibilities = 0;
        b_noise_aux = variables.noise;
        noise_cut = variables.noise_cut;
        random_probability = variables.randoms;
        eta = variables.eta;
        gridding = variables.gridding;
        nu_0 = variables.nu_0;
        robust_param = variables.robust_param;
        threshold = variables.threshold * 5.0;

        char *pt;
        char *temp = (char*)malloc(sizeof(char)*strlen(variables.initial_values));
        image_count = 0;
        strcpy(temp, variables.initial_values);
        pt = strtok(temp, ",");
        while(pt!=NULL) {
                image_count++;
                pt = strtok (NULL, ",");
        }
        free(pt);
        free(temp);

        if(image_count > 1 && nu_0 == -1)
        {
                print_help();
                printf("for 2 or more images, nu_0 (-F) is mandatory\n");
                exit(-1);
        }
        multigpu = 0;
        firstgpu = -1;

        struct stat st = {0};

        if(print_images)
                if(stat(mempath, &st) == -1) mkdir(mempath,0700);

        cudaDeviceProp dprop[num_gpus];
        if(verbose_flag) {
                printf("Number of host CPUs:\t%d\n", omp_get_num_procs());
                printf("Number of CUDA devices:\t%d\n", num_gpus);


                for(int i = 0; i < num_gpus; i++) {
                        cudaGetDeviceProperties(&dprop[i], i);

                        printf("> GPU%d = \"%15s\" %s capable of Peer-to-Peer (P2P)\n", i, dprop[i].name, (IsGPUCapableP2P(&dprop[i]) ? "IS " : "NOT"));

                        //printf("   %d: %s\n", i, dprop.name);
                }
                printf("---------------------------\n");
        }

        if(variables.blockSizeX*variables.blockSizeY >= dprop[0].maxThreadsPerBlock || variables.blockSizeV >= dprop[0].maxThreadsPerBlock){
            printf("ERROR. The maximum threads per block cannot be greater than %d\n", dprop[0].maxThreadsPerBlock);
            exit(-1);
        }

        if(variables.blockSizeX >= dprop[0].maxThreadsDim[0] || variables.blockSizeY >= dprop[0].maxThreadsDim[1] || variables.blockSizeV >= dprop[0].maxThreadsDim[0]){
          printf("ERROR. The size of the blocksize cannot exceed X: %d Y: %d Z: %d\n", dprop[0].maxThreadsDim[0], dprop[0].maxThreadsDim[1], dprop[0].maxThreadsDim[2]);
          exit(-1);
        }

        if(selected > num_gpus || selected < 0) {
                printf("ERROR. THE SELECTED GPU DOESN'T EXIST\n");
                exit(-1);
        }

        readInputDat(inputdat);
        init_beam(t_telescope);
        if(verbose_flag) {
                printf("Counting data for memory allocation\n");
        }

        canvasVariables canvas_vars = iohandler->IoreadCanvas(modinput, mod_in, b_noise_aux, status_mod_in, verbose_flag);

        M = canvas_vars.M;
        N = canvas_vars.N;
        DELTAX = canvas_vars.DELTAX;
        DELTAY = canvas_vars.DELTAY;
        ra = canvas_vars.ra;
        dec = canvas_vars.dec;
        crpix1 = canvas_vars.crpix1;
        crpix2 = canvas_vars.crpix2;
        beam_bmaj = canvas_vars.beam_bmaj;
        beam_bmin = canvas_vars.beam_bmin;
        beam_noise = canvas_vars.beam_noise;

        data = iohandler->IocountVisibilities(msinput, fields, gridding);

        vars_per_field = (VariablesPerField*)malloc(data.nfields*sizeof(VariablesPerField));

        if(verbose_flag) {
                printf("Number of fields = %d\n", data.nfields);
                printf("Number of frequencies = %d\n", data.total_frequencies);
        }

        if(strcmp(variables.multigpu, "NULL")!=0) {
                //Counts number of gpus to use
                char *pt;
                pt = strtok(variables.multigpu,",");

                while(pt!=NULL) {
                        if(multigpu==0) {
                                firstgpu = atoi(pt);
                        }
                        multigpu++;
                        pt = strtok(NULL, ",");
                }
        }else{
                multigpu = 0;
        }

        if(strcmp(variables.penalization_factors, "NULL")!=0) {
                int count = 0;
                char *pt;
                char *temp = (char*)malloc(sizeof(char)*strlen(variables.penalization_factors));
                strcpy(temp, variables.penalization_factors);
                pt = strtok(temp,",");
                while(pt!=NULL) {
                        count++;
                        pt = strtok(NULL, ",");
                }

                nPenalizators = count;

                strcpy(temp, variables.penalization_factors);
                pt = strtok(temp,",");
                penalizators = (float*)malloc(sizeof(float)*count);
                for(int i = 0; i < count; i++) {
                        penalizators[i] = atof(pt);
                        pt = strtok(NULL, ",");
                }

        }else{
                printf("no penalization factors provided\n");
        }

        if(multigpu < 0 || multigpu > num_gpus) {
                printf("ERROR. NUMBER OF GPUS CANNOT BE NEGATIVE OR GREATER THAN THE NUMBER OF GPUS\n");
                exit(-1);
        }else{
                if(multigpu == 0) {
                        num_gpus = 1;
                }else{
                        if(data.total_frequencies == 1) {
                                printf("ONLY ONE FREQUENCY. CHANGING NUMBER OF GPUS TO 1\n");
                                num_gpus = 1;
                        }else{
                                num_gpus = multigpu;
                                omp_set_num_threads(num_gpus);
                        }
                }
        }

        //printf("number of FINAL host CPUs:\t%d\n", omp_get_num_procs());
        if(verbose_flag) {
                printf("Number of CUDA devices and threads: \t%d\n", num_gpus);
        }

        //Check peer access if there is more than 1 GPU
        if(num_gpus > 1) {
                for(int i=firstgpu + 1; i< firstgpu + num_gpus; i++) {
                        cudaDeviceProp dprop0, dpropX;
                        cudaGetDeviceProperties(&dprop0, firstgpu);
                        cudaGetDeviceProperties(&dpropX, i);
                        int canAccessPeer0_x, canAccessPeerx_0;
                        cudaDeviceCanAccessPeer(&canAccessPeer0_x, firstgpu, i);
                        cudaDeviceCanAccessPeer(&canAccessPeerx_0, i, firstgpu);
                        if(verbose_flag) {
                                printf("> Peer-to-Peer (P2P) access from %s (GPU%d) -> %s (GPU%d) : %s\n", dprop0.name, firstgpu, dpropX.name, i, canAccessPeer0_x ? "Yes" : "No");
                                printf("> Peer-to-Peer (P2P) access from %s (GPU%d) -> %s (GPU%d) : %s\n", dpropX.name, i, dprop0.name, firstgpu, canAccessPeerx_0 ? "Yes" : "No");
                        }
                        if(canAccessPeer0_x == 0 || canAccessPeerx_0 == 0) {
                                printf("Two or more SM 2.0 class GPUs are required for %s to run.\n", argv[0]);
                                printf("Support for UVA requires a GPU with SM 2.0 capabilities.\n");
                                printf("Peer to Peer access is not available between GPU%d <-> GPU%d, waiving test.\n", 0, i);
                                exit(EXIT_SUCCESS);
                        }else{
                                cudaSetDevice(firstgpu);
                                if(verbose_flag) {
                                        printf("Granting access from %d to %d...\n",firstgpu, i);
                                }
                                cudaDeviceEnablePeerAccess(i,0);
                                cudaSetDevice(i);
                                if(verbose_flag) {
                                        printf("Granting access from %d to %d...\n", i, firstgpu);
                                }
                                cudaDeviceEnablePeerAccess(firstgpu,0);
                                if(verbose_flag) {
                                        printf("Checking GPU %d and GPU %d for UVA capabilities...\n", firstgpu, i);
                                }
                                const bool has_uva = (dprop0.unifiedAddressing && dpropX.unifiedAddressing);
                                if(verbose_flag) {
                                        printf("> %s (GPU%d) supports UVA: %s\n", dprop0.name, firstgpu, (dprop0.unifiedAddressing ? "Yes" : "No"));
                                        printf("> %s (GPU%d) supports UVA: %s\n", dpropX.name, i, (dpropX.unifiedAddressing ? "Yes" : "No"));
                                }
                                if (has_uva) {
                                        if(verbose_flag) {
                                                printf("Both GPUs can support UVA, enabling...\n");
                                        }
                                }
                                else{
                                        printf("At least one of the two GPUs does NOT support UVA, waiving test.\n");
                                        exit(EXIT_SUCCESS);
                                }
                        }
                }

                vars_gpu = (varsPerGPU*)malloc(num_gpus*sizeof(varsPerGPU));
        }

        for(int f=0; f<data.nfields; f++) {
                fields[f].visibilities = (Vis*)malloc(data.total_frequencies*sizeof(Vis));
                fields[f].gridded_visibilities = (Vis*)malloc(data.total_frequencies*sizeof(Vis));
                fields[f].device_visibilities = (Vis*)malloc(data.total_frequencies*sizeof(Vis));
                fields[f].backup_visibilities = (Vis*)malloc(data.total_frequencies*sizeof(Vis));
        }

        //ALLOCATE MEMORY AND GET TOTAL NUMBER OF VISIBILITIES
        for(int f=0; f<data.nfields; f++) {
                for(int i=0; i < data.total_frequencies; i++) {
                        fields[f].visibilities[i].stokes = (int*)malloc(fields[f].numVisibilitiesPerFreq[i]*sizeof(int));
                        fields[f].visibilities[i].u = (float*)malloc(fields[f].numVisibilitiesPerFreq[i]*sizeof(float));
                        fields[f].visibilities[i].v = (float*)malloc(fields[f].numVisibilitiesPerFreq[i]*sizeof(float));
                        fields[f].visibilities[i].weight = (float*)malloc(fields[f].numVisibilitiesPerFreq[i]*sizeof(float));
                        fields[f].visibilities[i].Vo = (cufftComplex*)malloc(fields[f].numVisibilitiesPerFreq[i]*sizeof(cufftComplex));
                        fields[f].visibilities[i].Vm = (cufftComplex*)malloc(fields[f].numVisibilitiesPerFreq[i]*sizeof(cufftComplex));

                        if(gridding)
                        {
                                fields[f].gridded_visibilities[i].u = (float*)malloc(M*N*sizeof(float));
                                fields[f].gridded_visibilities[i].v = (float*)malloc(M*N*sizeof(float));
                                fields[f].gridded_visibilities[i].weight = (float*)malloc(M*N*sizeof(float));
                                fields[f].gridded_visibilities[i].S = (int*)malloc(M*N*sizeof(int));
                                fields[f].gridded_visibilities[i].Vo = (cufftComplex*)malloc(M*N*sizeof(cufftComplex));
                                fields[f].gridded_visibilities[i].Vm = (cufftComplex*)malloc(M*N*sizeof(cufftComplex));

                                memset(fields[f].gridded_visibilities[i].u, 0, M*N*sizeof(float));
                                memset(fields[f].gridded_visibilities[i].v, 0, M*N*sizeof(float));
                                memset(fields[f].gridded_visibilities[i].weight, 0, M*N*sizeof(float));
                                memset(fields[f].gridded_visibilities[i].S, 0, M*N*sizeof(int));
                                memset(fields[f].gridded_visibilities[i].Vo, 0, M*N*sizeof(cufftComplex));
                                memset(fields[f].gridded_visibilities[i].Vm, 0, M*N*sizeof(cufftComplex));

                                //Save memory to backup original visibilities after gridding
                                fields[f].backup_visibilities[i].u = (float*)malloc(fields[f].numVisibilitiesPerFreq[i]*sizeof(float));
                                fields[f].backup_visibilities[i].v = (float*)malloc(fields[f].numVisibilitiesPerFreq[i]*sizeof(float));
                                fields[f].backup_visibilities[i].weight = (float*)malloc(fields[f].numVisibilitiesPerFreq[i]*sizeof(float));
                                fields[f].backup_visibilities[i].Vo = (cufftComplex*)malloc(fields[f].numVisibilitiesPerFreq[i]*sizeof(cufftComplex));
                                fields[f].backup_visibilities[i].Vm = (cufftComplex*)malloc(fields[f].numVisibilitiesPerFreq[i]*sizeof(cufftComplex));

                                memset(fields[f].backup_visibilities[i].u, 0, fields[f].numVisibilitiesPerFreq[i]*sizeof(float));
                                memset(fields[f].backup_visibilities[i].v, 0, fields[f].numVisibilitiesPerFreq[i]*sizeof(float));
                                memset(fields[f].backup_visibilities[i].weight, 0, fields[f].numVisibilitiesPerFreq[i]*sizeof(float));
                                memset(fields[f].backup_visibilities[i].Vo, 0, fields[f].numVisibilitiesPerFreq[i]*sizeof(cufftComplex));
                                memset(fields[f].backup_visibilities[i].Vm, 0, fields[f].numVisibilitiesPerFreq[i]*sizeof(cufftComplex));

                        }
                }
        }



        if(verbose_flag) {
                printf("Reading visibilities and FITS input files...\n");
        }



        if(apply_noise && random_probability < 1.0) {
                iohandler->IoreadMCNoiseSubsampledMS(msinput, fields, data, random_probability);
        }else if(random_probability < 1.0) {
                iohandler->IoreadSubsampledMS(msinput, fields, data, random_probability);
        }else if(apply_noise) {
                iohandler->IoreadMSMCNoise(msinput, fields, data);
        }else{
                iohandler->IoreadMS(msinput, fields, data);
        }
        this->visibilities = new Visibilities();
        this->visibilities->setData(&data);
        this->visibilities->setFields(fields);
        this->visibilities->setTotalVisibilites(&total_visibilities);
        float deltax = RPDEG*DELTAX; //radians
        float deltay = RPDEG*DELTAY; //radians
        deltau = 1.0 / (M * deltax);
        deltav = 1.0 / (N * deltay);

        if(gridding) {
                printf("Doing gridding\n");
                omp_set_num_threads(gridding);
                do_gridding(fields, &data, deltau, deltav, M, N, robust_param);
                omp_set_num_threads(num_gpus);
        }
}

void MFS::setDevice()
{
        float deltax = RPDEG*DELTAX; //radians
        float deltay = RPDEG*DELTAY; //radians
        deltau = 1.0 / (M * deltax);
        deltav = 1.0 / (N * deltay);

        if(verbose_flag) {
                printf("MS File Successfully Read\n");
                if(beam_noise == -1) {
                        printf("Beam noise wasn't provided by the user... Calculating...\n");
                }
        }

        sum_weights = calculateNoise(fields, data, &total_visibilities, variables.blockSizeV, gridding);

        if(num_gpus == 1) {
                cudaSetDevice(selected);
                for(int f=0; f<data.nfields; f++) {
                        for(int i=0; i<data.total_frequencies; i++) {
                                gpuErrchk(cudaMalloc(&fields[f].device_visibilities[i].u, sizeof(float)*fields[f].numVisibilitiesPerFreq[i]));
                                gpuErrchk(cudaMalloc(&fields[f].device_visibilities[i].v, sizeof(float)*fields[f].numVisibilitiesPerFreq[i]));
                                gpuErrchk(cudaMalloc(&fields[f].device_visibilities[i].Vo, sizeof(cufftComplex)*fields[f].numVisibilitiesPerFreq[i]));
                                gpuErrchk(cudaMalloc(&fields[f].device_visibilities[i].weight, sizeof(float)*fields[f].numVisibilitiesPerFreq[i]));
                                gpuErrchk(cudaMalloc(&fields[f].device_visibilities[i].Vm, sizeof(cufftComplex)*fields[f].numVisibilitiesPerFreq[i]));
                                gpuErrchk(cudaMalloc(&fields[f].device_visibilities[i].Vr, sizeof(cufftComplex)*fields[f].numVisibilitiesPerFreq[i]));
                        }
                }
        }else{
                for(int f=0; f<data.nfields; f++) {
                        for(int i=0; i<data.total_frequencies; i++) {
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


        if(num_gpus == 1) {
                cudaSetDevice(selected);
                gpuErrchk(cudaMalloc((void**)&device_dchi2, sizeof(float)*M*N));
                gpuErrchk(cudaMemset(device_dchi2, 0, sizeof(float)*M*N));

                gpuErrchk(cudaMalloc(&device_chi2, sizeof(float)*data.max_number_visibilities_in_channel));
                gpuErrchk(cudaMemset(device_chi2, 0, sizeof(float)*data.max_number_visibilities_in_channel));


                for(int f=0; f<data.nfields; f++) {
                        gpuErrchk(cudaMalloc((void**)&vars_per_field[f].atten_image, sizeof(float)*M*N));
                        gpuErrchk(cudaMemset(vars_per_field[f].atten_image, 0, sizeof(float)*M*N));
                        for(int i=0; i < data.total_frequencies; i++) {

                                //gpuErrchk(cudaMalloc(&vars_per_field[f].device_vars[i].chi2, sizeof(float)*fields[f].numVisibilitiesPerFreq[i]));
                                //gpuErrchk(cudaMemset(vars_per_field[f].device_vars[i].chi2, 0, sizeof(float)*fields[f].numVisibilitiesPerFreq[i]));

                                gpuErrchk(cudaMemcpy(fields[f].device_visibilities[i].u, fields[f].visibilities[i].u, sizeof(float)*fields[f].numVisibilitiesPerFreq[i], cudaMemcpyHostToDevice));

                                gpuErrchk(cudaMemcpy(fields[f].device_visibilities[i].v, fields[f].visibilities[i].v, sizeof(float)*fields[f].numVisibilitiesPerFreq[i], cudaMemcpyHostToDevice));

                                gpuErrchk(cudaMemcpy(fields[f].device_visibilities[i].weight, fields[f].visibilities[i].weight, sizeof(float)*fields[f].numVisibilitiesPerFreq[i], cudaMemcpyHostToDevice));

                                gpuErrchk(cudaMemcpy(fields[f].device_visibilities[i].Vo, fields[f].visibilities[i].Vo, sizeof(cufftComplex)*fields[f].numVisibilitiesPerFreq[i], cudaMemcpyHostToDevice));

                                gpuErrchk(cudaMemset(fields[f].device_visibilities[i].Vr, 0, sizeof(cufftComplex)*fields[f].numVisibilitiesPerFreq[i]));
                                gpuErrchk(cudaMemset(fields[f].device_visibilities[i].Vm, 0, sizeof(cufftComplex)*fields[f].numVisibilitiesPerFreq[i]));

                        }
                }
        }else{

                for(int g=0; g<num_gpus; g++) {
                        cudaSetDevice((g%num_gpus) + firstgpu);
                        gpuErrchk(cudaMalloc((void**)&vars_gpu[g].device_dchi2, sizeof(float)*M*N));
                        gpuErrchk(cudaMemset(vars_gpu[g].device_dchi2, 0, sizeof(float)*M*N));

                        gpuErrchk(cudaMalloc(&vars_gpu[g].device_chi2, sizeof(float)*data.max_number_visibilities_in_channel));
                        gpuErrchk(cudaMemset(vars_gpu[g].device_chi2, 0, sizeof(float)*data.max_number_visibilities_in_channel));
                }

                for(int f=0; f<data.nfields; f++) {
                        cudaSetDevice(firstgpu);
                        gpuErrchk(cudaMalloc((void**)&vars_per_field[f].atten_image, sizeof(float)*M*N));
                        gpuErrchk(cudaMemset(vars_per_field[f].atten_image, 0, sizeof(float)*M*N));
                        for(int i=0; i < data.total_frequencies; i++) {
                                cudaSetDevice((i%num_gpus) + firstgpu);
                                //gpuErrchk(cudaMalloc(&vars_per_field[f].device_vars[i].chi2, sizeof(float)*fields[f].numVisibilitiesPerFreq[i]));
                                //gpuErrchk(cudaMemset(vars_per_field[f].device_vars[i].chi2, 0, sizeof(float)*fields[f].numVisibilitiesPerFreq[i]));

                                //gpuErrchk(cudaMalloc((void**)&vars_per_field[f].device_vars[i].dchi2, sizeof(float)*M*N));
                                //gpuErrchk(cudaMemset(vars_per_field[f].device_vars[i].dchi2, 0, sizeof(float)*M*N));

                                gpuErrchk(cudaMemcpy(fields[f].device_visibilities[i].u, fields[f].visibilities[i].u, sizeof(float)*fields[f].numVisibilitiesPerFreq[i], cudaMemcpyHostToDevice));

                                gpuErrchk(cudaMemcpy(fields[f].device_visibilities[i].v, fields[f].visibilities[i].v, sizeof(float)*fields[f].numVisibilitiesPerFreq[i], cudaMemcpyHostToDevice));

                                gpuErrchk(cudaMemcpy(fields[f].device_visibilities[i].weight, fields[f].visibilities[i].weight, sizeof(float)*fields[f].numVisibilitiesPerFreq[i], cudaMemcpyHostToDevice));

                                gpuErrchk(cudaMemcpy(fields[f].device_visibilities[i].Vo, fields[f].visibilities[i].Vo, sizeof(cufftComplex)*fields[f].numVisibilitiesPerFreq[i], cudaMemcpyHostToDevice));

                                gpuErrchk(cudaMemset(fields[f].device_visibilities[i].Vr, 0, sizeof(cufftComplex)*fields[f].numVisibilitiesPerFreq[i]));
                                gpuErrchk(cudaMemset(fields[f].device_visibilities[i].Vm, 0, sizeof(cufftComplex)*fields[f].numVisibilitiesPerFreq[i]));
                        }
                }
        }

        //Declaring block size and number of blocks for Image
        dim3 threads(variables.blockSizeX, variables.blockSizeY);
        dim3 blocks(M/threads.x, N/threads.y);
        threadsPerBlockNN = threads;
        numBlocksNN = blocks;

        noise_jypix = beam_noise / (PI * beam_bmaj * beam_bmin / (4 * log(2) ));

        /////////////////////////////////////////////////////CALCULATE DIRECTION COSINES/////////////////////////////////////////////////
        double raimage = ra * RPDEG_D;
        double decimage = dec * RPDEG_D;

        if(verbose_flag) {
                printf("FITS: Ra: %.16e (rad), dec: %.16e (rad)\n", raimage, decimage);
                printf("FITS: Center pix: (%d,%d)\n", crpix1-1, crpix2-1);
        }

        double lobs, mobs, lphs, mphs;
        double dcosines_l_pix_ref, dcosines_m_pix_ref, dcosines_l_pix_phs, dcosines_m_pix_phs;
        for(int f=0; f<data.nfields; f++) {

                direccos(fields[f].ref_ra, fields[f].ref_dec, raimage, decimage, &lobs,  &mobs);
                direccos(fields[f].phs_ra, fields[f].phs_dec, raimage, decimage, &lphs,  &mphs);

                dcosines_l_pix_ref = lobs/deltax; // Radians to pixels
                dcosines_m_pix_ref = mobs/deltay; // Radians to pixels

                dcosines_l_pix_phs = lphs/deltax; // Radians to pixels
                dcosines_m_pix_phs = mphs/deltay; // Radians to pixels
                if(verbose_flag)
                {
                    printf("Ref: l (pix): %e, m (pix): %e\n", dcosines_l_pix_ref, dcosines_m_pix_ref);
                    printf("Phase: l (pix): %e, m (pix): %e\n", dcosines_l_pix_phs, dcosines_m_pix_phs);

                }

                if(crpix1 != crpix2) {
                    fields[f].ref_xobs = (crpix1 - 1.0f) + dcosines_l_pix_ref;// + 1.0f;// + 6.0f;
                    fields[f].ref_yobs = (crpix2 - 1.0f) + dcosines_m_pix_ref;// - 1.0f;// - 10.0f;

                    fields[f].phs_xobs = (crpix1 - 1.0f) + dcosines_l_pix_phs;// + 1.0f;// + 6.0f;
                    fields[f].phs_yobs = (crpix2 - 1.0f) + dcosines_m_pix_phs;// - 1.0f;// - 10.0f;
                }else{
                    fields[f].ref_xobs = (crpix1 - 1.0f) + dcosines_l_pix_ref; //- 1.0f;// + 6.0f;
                    fields[f].ref_yobs = (crpix2 - 1.0f) + dcosines_m_pix_ref; //- 1.0f;// - 7.0f;

                    fields[f].phs_xobs = (crpix1 - 1.0f) + dcosines_l_pix_phs; //- 1.0f;// + 5.0f;
                    fields[f].phs_yobs = (crpix2 - 1.0f) + dcosines_m_pix_phs; //- 1.0f;// - 7.0f;
                }

                if(verbose_flag) {
                    printf("Ref: Field %d - Ra: %.16e (rad), dec: %.16e (rad), x0: %f (pix), y0: %f (pix)\n", f, fields[f].ref_ra, fields[f].ref_dec,
                           fields[f].ref_xobs, fields[f].ref_yobs);
                    printf("Phase: Field %d - Ra: %.16e (rad), dec: %.16e (rad), x0: %f (pix), y0: %f (pix)\n", f, fields[f].phs_ra, fields[f].phs_dec,
                           fields[f].phs_xobs, fields[f].phs_yobs);
                }

                if(fields[f].ref_xobs < 0 || fields[f].ref_xobs >= M || fields[f].ref_xobs < 0 || fields[f].ref_yobs >= N) {
                        printf("Pointing reference center (%f,%f) is outside the range of the image\n", fields[f].ref_xobs, fields[f].ref_yobs);
                        goToError();
                }

                if(fields[f].phs_xobs < 0 || fields[f].phs_xobs >= M || fields[f].phs_xobs < 0 || fields[f].phs_yobs >= N) {
                    printf("Pointing phase center (%f,%f) is outside the range of the image\n", fields[f].phs_xobs, fields[f].phs_yobs);
                    goToError();
                }
        }
        ////////////////////////////////////////////////////////MAKE STARTING IMAGE////////////////////////////////////////////////////////

        char *pt;
        char *temp = (char*)malloc(sizeof(char)*strlen(variables.initial_values));
        strcpy(temp, variables.initial_values);
        if(image_count == 1) {
                initial_values = (float*)malloc(sizeof(float)*image_count+1);
        }else{
                initial_values = (float*)malloc(sizeof(float)*image_count);
        }
        pt = strtok(temp, ",");
        for(int i=0; i< image_count; i++) {
                initial_values[i] = atof(pt);
                pt = strtok (NULL, ",");
        }

        if(image_count == 1)
        {
                initial_values[1] = 0.0f;
                image_count++;
                nu_0 = 1.0f;
                imagesChanged = 1;
        }

        host_I = (float*)malloc(M*N*sizeof(float)*image_count);

        free(pt);
        free(temp);
        for(int i=0; i<M; i++) {
                for(int j=0; j<N; j++) {
                        for(int k=0; k<image_count; k++) {
                                host_I[N*M*k+N*i+j] = initial_values[k];
                        }
                }
        }

        ////////////////////////////////////////////////CUDA MEMORY ALLOCATION FOR DEVICE///////////////////////////////////////////////////

        if(num_gpus == 1) {
                cudaSetDevice(selected);
                gpuErrchk(cudaMalloc((void**)&device_V, sizeof(cufftComplex)*M*N));
                gpuErrchk(cudaMalloc((void**)&device_image, sizeof(cufftComplex)*M*N));
        }else{
                for(int g=0; g<num_gpus; g++) {
                        cudaSetDevice((g%num_gpus) + firstgpu);
                        gpuErrchk(cudaMalloc((void**)&vars_gpu[g].device_V, sizeof(cufftComplex)*M*N));
                        gpuErrchk(cudaMalloc((void**)&vars_gpu[g].device_image, sizeof(cufftComplex)*M*N));
                }
        }

        if(num_gpus == 1) {
                cudaSetDevice(selected);
        }else{
                cudaSetDevice(firstgpu);
        }
        gpuErrchk(cudaMalloc((void**)&device_Image, sizeof(float)*M*N*image_count));
        gpuErrchk(cudaMemset(device_Image, 0, sizeof(float)*M*N*image_count));

        gpuErrchk(cudaMemcpy(device_Image, host_I, sizeof(float)*N*M*image_count, cudaMemcpyHostToDevice));

        gpuErrchk(cudaMalloc((void**)&device_noise_image, sizeof(float)*M*N));
        gpuErrchk(cudaMemset(device_noise_image, 0, sizeof(float)*M*N));

        gpuErrchk(cudaMalloc((void**)&device_weight_image, sizeof(float)*M*N));
        gpuErrchk(cudaMemset(device_weight_image, 0, sizeof(float)*M*N));


        if(num_gpus == 1) {
                cudaSetDevice(selected);
                gpuErrchk(cudaMemset(device_V, 0, sizeof(cufftComplex)*M*N));
                gpuErrchk(cudaMemset(device_image, 0, sizeof(cufftComplex)*M*N));
        }else{
                for(int g=0; g<num_gpus; g++) {
                        cudaSetDevice((g%num_gpus) + firstgpu);
                        gpuErrchk(cudaMemset(vars_gpu[g].device_V, 0, sizeof(cufftComplex)*M*N));
                        gpuErrchk(cudaMemset(vars_gpu[g].device_image, 0, sizeof(cufftComplex)*M*N));
                }
        }

        /////////// MAKING IMAGE OBJECT /////////////
        image = new Image(device_Image, image_count);
        imageMap *functionPtr = (imageMap*)malloc(sizeof(imageMap)*image_count);
        image->setFunctionMapping(functionPtr);

        for(int i = 0; i < image_count; i++)
        {
                if(nopositivity)
                {
                        functionPtr[i].evaluateXt = defaultEvaluateXt;
                        functionPtr[i].newP = defaultNewP;
                }else{
                        if(!i)
                        {
                              functionPtr[i].evaluateXt = particularEvaluateXt;
                              functionPtr[i].newP = particularNewP;
                        }else{
                              functionPtr[i].evaluateXt = defaultEvaluateXt;
                              functionPtr[i].newP = defaultNewP;
                        }
                }
        }

        if(num_gpus == 1) {
                cudaSetDevice(selected);
                if ((cufftPlan2d(&plan1GPU, N, M, CUFFT_C2C))!= CUFFT_SUCCESS) {
                        printf("cufft plan error\n");
                        exit(-1);
                }
        }else{
                for(int g=0; g<num_gpus; g++) {
                        cudaSetDevice((g%num_gpus) + firstgpu);
                        if ((cufftPlan2d(&vars_gpu[g].plan, N, M, CUFFT_C2C))!= CUFFT_SUCCESS) {
                                printf("cufft plan error\n");
                                exit(-1);
                        }
                }
        }

        //Time is taken from first kernel
        t = clock();
        start = omp_get_wtime();
        if(num_gpus == 1) {
                cudaSetDevice(selected);
                for(int f=0; f < data.nfields; f++) {
                        for(int i=0; i<data.total_frequencies; i++) {
                                hermitianSymmetry<<<fields[f].visibilities[i].numBlocksUV, fields[f].visibilities[i].threadsPerBlockUV>>>(fields[f].device_visibilities[i].u, fields[f].device_visibilities[i].v, fields[f].device_visibilities[i].Vo, fields[f].visibilities[i].freq, fields[f].numVisibilitiesPerFreq[i]);
                                gpuErrchk(cudaDeviceSynchronize());
                        }
                }
        }else{
                for(int f = 0; f < data.nfields; f++) {
      #pragma omp parallel for schedule(static,1)
                        for (int i = 0; i < data.total_frequencies; i++)
                        {
                                unsigned int j = omp_get_thread_num();
                                //unsigned int num_cpu_threads = omp_get_num_threads();
                                // set and check the CUDA device for this CPU thread
                                int gpu_id = -1;
                                cudaSetDevice((i%num_gpus) + firstgpu); // "% num_gpus" allows more CPU threads than GPU devices
                                cudaGetDevice(&gpu_id);
                                hermitianSymmetry<<<fields[f].visibilities[i].numBlocksUV, fields[f].visibilities[i].threadsPerBlockUV>>>(fields[f].device_visibilities[i].u, fields[f].device_visibilities[i].v, fields[f].device_visibilities[i].Vo, fields[f].visibilities[i].freq, fields[f].numVisibilitiesPerFreq[i]);
                                gpuErrchk(cudaDeviceSynchronize());
                        }

                }
        }

        if(num_gpus == 1) {
                cudaSetDevice(selected);
                for(int f=0; f<data.nfields; f++) {
                        for(int i=0; i<data.total_frequencies; i++) {
                                if(fields[f].numVisibilitiesPerFreq[i] > 0) {
                                        total_attenuation<<<numBlocksNN, threadsPerBlockNN>>>(vars_per_field[f].atten_image, antenna_diameter, pb_factor, pb_cutoff, fields[f].visibilities[i].freq, fields[f].ref_xobs, fields[f].ref_yobs, DELTAX, DELTAY, N);
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
                                if(fields[f].numVisibilitiesPerFreq[i] > 0) {
                                        #pragma omp critical
                                        {
                                                total_attenuation<<<numBlocksNN, threadsPerBlockNN>>>(vars_per_field[f].atten_image, antenna_diameter, pb_factor, pb_cutoff, fields[f].visibilities[i].freq, fields[f].ref_xobs, fields[f].ref_yobs, DELTAX, DELTAY, N);
                                                gpuErrchk(cudaDeviceSynchronize());
                                        }
                                }
                        }
                }
        }

        for(int f=0; f<data.nfields; f++) {
                if(fields[f].valid_frequencies > 0) {
                        if(num_gpus == 1) {
                                cudaSetDevice(selected);
                                mean_attenuation<<<numBlocksNN, threadsPerBlockNN>>>(vars_per_field[f].atten_image, fields[f].valid_frequencies, N);
                                gpuErrchk(cudaDeviceSynchronize());
                        }else{
                                cudaSetDevice(firstgpu);
                                mean_attenuation<<<numBlocksNN, threadsPerBlockNN>>>(vars_per_field[f].atten_image, fields[f].valid_frequencies, N);
                                gpuErrchk(cudaDeviceSynchronize());
                        }
                        if(print_images)
                                iohandler->IoPrintImageIteration(vars_per_field[f].atten_image, mod_in, mempath, "atten", "", f, 0, 0.0, M, N);
                }
        }

        if(num_gpus == 1) {
                cudaSetDevice(selected);
        }else{
                cudaSetDevice(firstgpu);
        }

        for(int f=0; f<data.nfields; f++) {
                weight_image<<<numBlocksNN, threadsPerBlockNN>>>(device_weight_image, vars_per_field[f].atten_image, noise_jypix, N);
                gpuErrchk(cudaDeviceSynchronize());
        }
        noise_image<<<numBlocksNN, threadsPerBlockNN>>>(device_noise_image, device_weight_image, noise_jypix, N);
        gpuErrchk(cudaDeviceSynchronize());
        if(print_images)
                iohandler->IoPrintImage(device_noise_image, mod_in, mempath, "noise.fits", "", 0, 0, 0.0, M, N);


        float *host_noise_image = (float*)malloc(M*N*sizeof(float));
        gpuErrchk(cudaMemcpy2D(host_noise_image, sizeof(float), device_noise_image, sizeof(float), sizeof(float), M*N, cudaMemcpyDeviceToHost));
        float noise_min = *std::min_element(host_noise_image,host_noise_image+(M*N));

        fg_scale = noise_min;
        noise_cut = noise_cut * noise_min;
        if(verbose_flag) {
                printf("fg_scale = %e\n", fg_scale);
                printf("noise (Jy/pix) = %e\n", noise_jypix);
        }
        free(host_noise_image);
        cudaFree(device_weight_image);
        for(int f=0; f<data.nfields; f++) {
                cudaFree(vars_per_field[f].atten_image);
        }
};

void MFS::run()
{
        //printf("\n\nStarting Fletcher Reeves Polak Ribiere method (Conj. Grad.)\n\n");
        printf("\n\nStarting Optimizator\n");
        optimizator->getObjectiveFuntion()->setIo(iohandler);
        optimizator->getObjectiveFuntion()->setPrintImages(print_images);
        //optimizator->getObjectiveFuntion()->setIoOrderIterations(IoOrderIterations);
        if(this->Order == NULL) {
                if(imagesChanged)
                {
                        optimizator->setImage(image);
                        optimizator->minimizate();
                }else if(image_count == 2) {
                        optimizator->setImage(image);
                        optimizator->setFlag(0);
                        optimizator->minimizate();
                        optimizator->setFlag(1);
                        optimizator->minimizate();
                        optimizator->setFlag(2);
                        optimizator->minimizate();
                        optimizator->setFlag(3);
                        optimizator->minimizate();
                }
        }else{
                (this->Order)(optimizator, image);
        }

        t = clock() - t;
        end = omp_get_wtime();
        printf("Minimization ended successfully\n\n");
        printf("Iterations: %d\n", iter);
        printf("chi2: %f\n", final_chi2);
        printf("0.5*chi2: %f\n", 0.5*final_chi2);
        printf("Total visibilities: %d\n", total_visibilities);
        printf("Reduced-chi2 (Num visibilities): %f\n", (0.5*final_chi2)/total_visibilities);
        printf("Reduced-chi2 (Weights sum): %f\n", (0.5*final_chi2)/sum_weights);
        printf("S: %f\n", final_S);
        if(reg_term != 1) {
                printf("Normalized S: %f\n", final_S/(M*N));
        }else{
                printf("Normalized S: %f\n", final_S/(M*M*N*N));
        }
        printf("lambda*S: %f\n\n", lambda*final_S);
        double time_taken = ((double)t)/CLOCKS_PER_SEC;
        double wall_time = end-start;
        printf("Total CPU time: %lf\n", time_taken);
        printf("Wall time: %lf\n\n\n", wall_time);

        if(strcmp(variables.ofile,"NULL") != 0) {
                FILE *outfile = fopen(variables.ofile, "w");
                if (outfile == NULL)
                {
                        printf("Error opening output file!\n");
                        goToError();
                }

                fprintf(outfile, "Iterations: %d\n", iter);
                fprintf(outfile, "chi2: %f\n", final_chi2);
                fprintf(outfile, "0.5*chi2: %f\n", 0.5*final_chi2);
                fprintf(outfile, "Total visibilities: %d\n", total_visibilities);
                fprintf(outfile, "Reduced-chi2 (Num visibilities): %f\n", (0.5*final_chi2)/total_visibilities);
                fprintf(outfile, "Reduced-chi2 (Weights sum): %f\n", (0.5*final_chi2)/sum_weights);
                fprintf(outfile, "S: %f\n", final_S);
                if(reg_term != 1) {
                        fprintf(outfile, "Normalized S: %f\n", final_S/(M*N));
                }else{
                        fprintf(outfile, "Normalized S: %f\n", final_S/(M*M*N*N));
                }
                fprintf(outfile, "lambda*S: %f\n", lambda*final_S);
                fprintf(outfile, "Wall time: %lf", wall_time);
                fclose(outfile);
        }
        //Pass residuals to host
        printf("Saving final image to disk\n");
        if(IoOrderEnd == NULL) {
                iohandler->IoPrintImage(image->getImage(), mod_in, "", out_image, "JY/PIXEL", iter, 0, fg_scale, M, N);
                iohandler->IoPrintImage(image->getImage(), mod_in, "", "alpha.fits", "", iter, 1, 0.0, M, N);
        }else{
                (IoOrderEnd)(image->getImage(), iohandler);
        }

        if(print_errors) /* flag for print error image */
        {
                if(this->error == NULL)
                {
                        this->error = Singleton<ErrorFactory>::Instance().CreateError(0);
                }
                /* code for calculate error */
                /* make void * params */
                printf("Calculating Error Images\n");
                this->error->calculateErrorImage(this->image, this->visibilities);
                if(IoOrderError == NULL) {
                        iohandler->IoPrintImage(image->getErrorImage(), mod_in, "", "error_Inu_0.fits", "JY/PIXEL", iter, 0, 0.0, M, N);
                        iohandler->IoPrintImage(image->getErrorImage(), mod_in, "", "error_alpha.fits", "", iter, 1, 0.0, M, N);
                }else{
                        (IoOrderError)(image->getErrorImage(), iohandler);
                }

        }

        if(!gridding)
        {
                //Saving residuals to disk
                residualsToHost(fields, data, num_gpus, firstgpu);
                printf("Saving residuals to MS...\n");
                iohandler->IowriteMS(msinput, msoutput, fields, data, random_probability, verbose_flag);
                printf("Residuals saved.\n");
        }else{
            float deltax = RPDEG*DELTAX; //radians
            float deltay = RPDEG*DELTAY; //radians
            deltau = 1.0 / (M * deltax);
            deltav = 1.0 / (N * deltay);

            printf("Visibilities are gridded, we will need to de-grid to save them in a Measurement Set File\n");
            degridding(fields, data, deltau, deltav, num_gpus, firstgpu, variables.blockSizeV, M, N);
            residualsToHost(fields, data, num_gpus, firstgpu);
            printf("Saving residuals to MS...\n");
            iohandler->IowriteMS(msinput, msoutput, fields, data, random_probability, verbose_flag);
            printf("Residuals saved.\n");
        }


};

void MFS::unSetDevice()
{
        //Free device and host memory
        printf("Freeing device memory\n");
        if(num_gpus == 1) {
            cudaSetDevice(selected);
        }else{
            cudaSetDevice(firstgpu);
        }

        //Weird error segmentation fault when doing cudafree
        for(int f=0; f<data.nfields; f++) {
            for(int i=0; i<data.total_frequencies; i++) {

                    if(num_gpus > 1) {
                        cudaSetDevice((i%num_gpus) + firstgpu);
                    }

                    cudaFree(fields[f].device_visibilities[i].u);
                    cudaFree(fields[f].device_visibilities[i].v);
                    cudaFree(fields[f].device_visibilities[i].weight);
                    cudaFree(fields[f].device_visibilities[i].Vr);
                    cudaFree(fields[f].device_visibilities[i].Vm);
                    cudaFree(fields[f].device_visibilities[i].Vo);

                }
        }

        printf("Freeing cuFFT plans\n");
        if(num_gpus > 1) {
                for(int g=0; g<num_gpus; g++) {
                        cudaSetDevice((g%num_gpus) + firstgpu);
                        cufftDestroy(vars_gpu[g].plan);
                }
        }else{
            cudaSetDevice(selected);
            cufftDestroy(plan1GPU);
        }

        printf("Freeing host memory\n");
        for(int f=0; f<data.nfields; f++) {
                for(int i=0; i<data.total_frequencies; i++) {
                        if(fields[f].numVisibilitiesPerFreq[i] != 0) {
                                free(fields[f].visibilities[i].u);
                                free(fields[f].visibilities[i].v);
                                free(fields[f].visibilities[i].weight);
                                free(fields[f].visibilities[i].Vo);
                                free(fields[f].visibilities[i].Vm);
                        }
                }
        }

        cudaFree(device_Image);
        if(num_gpus == 1) {
                cudaFree(device_V);
                cudaFree(device_image);
        }else{
                for(int g=0; g<num_gpus; g++) {
                        cudaSetDevice((g%num_gpus) + firstgpu);
                        cudaFree(vars_gpu[g].device_V);
                        cudaFree(vars_gpu[g].device_image);
                }
        }
        if(num_gpus == 1) {
                cudaSetDevice(selected);
        }else{
                cudaSetDevice(firstgpu);
        }

        cudaFree(device_noise_image);
        cudaFree(device_fg_image);

        cudaFree(device_dphi);
        cudaFree(device_dchi2);
        cudaFree(device_chi2);
        cudaFree(device_dchi2_total);
        cudaFree(device_dS);

        cudaFree(device_S);

        //Disabling UVA
        if(num_gpus > 1) {
                for(int i=firstgpu+1; i<num_gpus+firstgpu; i++) {
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
        free(host_I);
        free(msinput);
        free(msoutput);
        free(modinput);

        iohandler->IocloseCanvas(mod_in);
};

namespace {
Synthesizer* CreateMFS()
{
        return new MFS;
}
const int MFSID = 0;
const bool RegisteredMFS = Singleton<SynthesizerFactory>::Instance().RegisterSynthesizer(MFSID, CreateMFS);
};
