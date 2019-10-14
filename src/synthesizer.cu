#include "synthesizer.cuh"
#include "imageProcessor.cuh"


long M, N, numVisibilities;
int iter=0;

float *device_Image, *device_dphi, *device_dchi2_total, *device_dS, *device_S, beam_noise, beam_bmaj, *device_noise_image, *device_weight_image;
float beam_bmin, b_noise_aux, noise_cut, MINPIX, minpix, lambda, ftol, random_probability = 1.0;
float noise_jypix, fg_scale, final_chi2, final_S, eta, robust_param;
float *host_I, sum_weights, *initial_values, *penalizators;
Telescope *telescope;

dim3 threadsPerBlockNN;
dim3 numBlocksNN;

int threadsVectorReduceNN, blocksVectorReduceNN, nopositivity = 0, verbose_flag = 0, clip_flag = 0, apply_noise = 0, print_images = 0, gridding, it_maximum, status_mod_in;
int firstgpu, selected, reg_term, total_visibilities, image_count, nPenalizators, print_errors, nMeasurementSets=0, max_number_vis;
char *output, *mempath, *out_image, *msinput, *msoutput, *inputdat, *modinput;
char *t_telescope;
float nu_0, threshold;
extern int num_gpus;

double ra, dec, crpix1, crpix2, DELTAX, DELTAY, deltau, deltav;

fitsfile *mod_in;

MSDataset *datasets;

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

std::vector<std::string> MFS::countAndSeparateStrings(char *input)
{
        char *pt;
        std::vector<std::string> ret;

        int counter = 0;
        pt = strtok(input, ",");
        while(pt!=NULL) {
                std::string s(pt);
                ret.push_back(s);
                pt = strtok (NULL, ",");
        }

        free(pt);
        return ret;
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
        std::vector<std::string> string_values;
        std::vector<std::string> s_output_values;
        int n_outputs;


        if(strcmp(msinput, "NULL")!=0) {
                string_values = countAndSeparateStrings(msinput);
                nMeasurementSets = string_values.size();
        }else{
                printf("Datasets files were not provided\n");
                print_help();
                exit(-1);
        }

        if(strcmp(msoutput, "NULL")!=0) {
                s_output_values = countAndSeparateStrings(msoutput);
                n_outputs = s_output_values.size();
        }else{
                printf("Output/s was/were not provided\n");
                print_help();
                exit(-1);
        }

        if(n_outputs != nMeasurementSets) {
                printf("Number of input datasets should be equal to the number of output datasets\n");
                exit(-1);
        }

        datasets = (MSDataset*)malloc(nMeasurementSets*sizeof(MSDataset));

        if(verbose_flag)
                printf("Number of input datasets %d\n", nMeasurementSets);

        for(int i=0; i< nMeasurementSets; i++) {

                datasets[i].name = (char*)malloc((string_values[i].length()+1)*sizeof(char));
                datasets[i].oname = (char*)malloc((s_output_values[i].length()+1)*sizeof(char));
                strcpy(datasets[i].name, string_values[i].c_str());
                strcpy(datasets[i].oname, s_output_values[i].c_str());
        }

        string_values.clear();
        s_output_values.clear();

        if(strcmp(variables.initial_values, "NULL")!=0) {
                string_values = countAndSeparateStrings(variables.initial_values);
                image_count = string_values.size();
        }else{
                printf("Initial values for image/s were not provided\n");
                print_help();
                exit(-1);
        }

        initial_values = (float*)malloc(sizeof(float)*image_count);

        for(int i=0; i< image_count; i++)
                initial_values[i] = atof(string_values[i].c_str());

        string_values.clear();
        nu_0 = 1.0f;
        imagesChanged = 1;

        /*if(image_count > 1 && nu_0 == -1)
           {
                print_help();
                printf("for 2 or more images, nu_0 (-F) is mandatory\n");
                exit(-1);
           }*/



        /*
         *
         * Create directory to save images for each iterations
         */
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

        cudaGetDeviceProperties(&dprop[0], 0);
        if(variables.blockSizeX*variables.blockSizeY > dprop[0].maxThreadsPerBlock || variables.blockSizeV > dprop[0].maxThreadsPerBlock) {
                printf("Block size X: %d\n", variables.blockSizeX);
                printf("Block size Y: %d\n", variables.blockSizeY);
                printf("Block size X*Y: %d\n", variables.blockSizeX*variables.blockSizeY);
                printf("Block size V: %d\n", variables.blockSizeV);
                printf("ERROR. The maximum threads per block cannot be greater than %d\n", dprop[0].maxThreadsPerBlock);
                exit(-1);
        }

        if(variables.blockSizeX > dprop[0].maxThreadsDim[0] || variables.blockSizeY > dprop[0].maxThreadsDim[1] || variables.blockSizeV > dprop[0].maxThreadsDim[0]) {
                printf("Block size X: %d\n", variables.blockSizeX);
                printf("Block size Y: %d\n", variables.blockSizeY);
                printf("Block size V: %d\n", variables.blockSizeV);
                printf("ERROR. The size of the blocksize cannot exceed X: %d Y: %d Z: %d\n", dprop[0].maxThreadsDim[0], dprop[0].maxThreadsDim[1], dprop[0].maxThreadsDim[2]);
                exit(-1);
        }

        if(selected > num_gpus || selected < 0) {
                printf("ERROR. THE SELECTED GPU DOESN'T EXIST\n");
                exit(-1);
        }

        readInputDat(inputdat);

        int n_telescopes;
        if(strcmp(t_telescope, "NULL")!=0) {
                string_values = countAndSeparateStrings(t_telescope);
                n_telescopes = string_values.size();
        }else{
                printf("Telescope codes were not provided\n");
                print_help();
                exit(-1);
        }

        if(n_telescopes != nMeasurementSets) {
                printf("Number of telescope codes cannot be different to the number of Measurement Sets\n");
                print_help();
                exit(-1);
        }

        int telescope;
        for(int i=0; i< nMeasurementSets; i++) {

                telescope = atoi(string_values[i].c_str());
                init_beam(telescope, &datasets[i].antenna_diameter, &datasets[i].pb_factor, &datasets[i].pb_cutoff);
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

        if(verbose_flag)
                printf("Counting data for memory allocation\n");

        for(int i=0; i<nMeasurementSets; i++) {
                datasets[i].data = iohandler->IocountVisibilities(datasets[i].name, datasets[i].fields, gridding);
        }

        if(verbose_flag) {
                for(int i=0; i<nMeasurementSets; i++) {
                        printf("Dataset %s\n", datasets[i].name);
                        printf("\tNumber of fields = %d\n", datasets[i].data.nfields);
                        printf("\tNumber of frequencies = %d\n", datasets[i].data.total_frequencies);
                        printf("\tNumber of Stokes = %d\n", datasets[i].data.nstokes);
                }
        }

        if(strcmp(variables.penalization_factors, "NULL")!=0) {

                string_values = countAndSeparateStrings(variables.penalization_factors);
                nPenalizators = string_values.size();
                penalizators = (float*)malloc(sizeof(float)*nPenalizators);
                for(int i = 0; i < nPenalizators; i++)
                        penalizators[i] = atof(string_values[i].c_str());

        }else{
                printf("No regularization factors provided\n");
        }
        string_values.clear();

        num_gpus = 1;
        firstgpu = selected;
        //printf("number of FINAL host CPUs:\t%d\n", omp_get_num_procs());
        if(verbose_flag) {
                printf("Number of CUDA devices and threads: \t%d\n", num_gpus);
        }


        vars_gpu = (varsPerGPU*)malloc(num_gpus*sizeof(varsPerGPU));

        for(int d=0; d<nMeasurementSets; d++) {
                for(int f=0; f<datasets[d].data.nfields; f++) {
                        datasets[d].fields[f].visibilities = (Vis**)malloc(datasets[d].data.total_frequencies*sizeof(Vis*));
                        datasets[d].fields[f].device_visibilities = (Vis**)malloc(datasets[d].data.total_frequencies*sizeof(Vis*));
                        datasets[d].fields[f].nu = (float*)malloc(datasets[d].data.total_frequencies*sizeof(float));
                        if(gridding) {
                                datasets[d].fields[f].gridded_visibilities = (Vis **) malloc(datasets[d].data.total_frequencies * sizeof(Vis * ));
                                datasets[d].fields[f].backup_visibilities = (Vis **) malloc(datasets[d].data.total_frequencies * sizeof(Vis * ));
                        }

                        for(int i=0; i<datasets[d].data.total_frequencies; i++) {
                                datasets[d].fields[f].visibilities[i] = (Vis*)malloc(datasets[d].data.nstokes*sizeof(Vis));
                                datasets[d].fields[f].device_visibilities[i] = (Vis*)malloc(datasets[d].data.nstokes*sizeof(Vis));

                                if(gridding) {
                                        datasets[d].fields[f].gridded_visibilities[i] = (Vis*)malloc(datasets[d].data.nstokes*sizeof(Vis));
                                        datasets[d].fields[f].backup_visibilities[i] = (Vis*)malloc(datasets[d].data.nstokes*sizeof(Vis));
                                }
                        }
                }
        }


        //ALLOCATE MEMORY AND GET TOTAL NUMBER OF VISIBILITIES
        for(int d=0; d<nMeasurementSets; d++) {
                for(int f=0; f<datasets[d].data.nfields; f++) {
                        for(int i=0; i < datasets[d].data.total_frequencies; i++) {
                                for(int s=0; s<datasets[d].data.nstokes; s++) {
                                        datasets[d].fields[f].visibilities[i][s].uvw = (double3*)malloc(datasets[d].fields[f].numVisibilitiesPerFreqPerStoke[i][s]*sizeof(double3));
                                        datasets[d].fields[f].visibilities[i][s].weight = (float*)malloc(datasets[d].fields[f].numVisibilitiesPerFreqPerStoke[i][s]*sizeof(float));
                                        datasets[d].fields[f].visibilities[i][s].Vo = (cufftComplex*)malloc(datasets[d].fields[f].numVisibilitiesPerFreqPerStoke[i][s]*sizeof(cufftComplex));
                                        datasets[d].fields[f].visibilities[i][s].Vm = (cufftComplex*)malloc(datasets[d].fields[f].numVisibilitiesPerFreqPerStoke[i][s]*sizeof(cufftComplex));

                                        if(gridding)
                                        {
                                                datasets[d].fields[f].gridded_visibilities[i][s].uvw = (double3*)malloc(M*N*sizeof(double3));
                                                datasets[d].fields[f].gridded_visibilities[i][s].weight = (float*)malloc(M*N*sizeof(float));
                                                datasets[d].fields[f].gridded_visibilities[i][s].S = (int*)malloc(M*N*sizeof(int));
                                                datasets[d].fields[f].gridded_visibilities[i][s].Vo = (cufftComplex*)malloc(M*N*sizeof(cufftComplex));
                                                datasets[d].fields[f].gridded_visibilities[i][s].Vm = (cufftComplex*)malloc(M*N*sizeof(cufftComplex));

                                                memset(datasets[d].fields[f].gridded_visibilities[i][s].uvw, 0, M*N*sizeof(double3));
                                                memset(datasets[d].fields[f].gridded_visibilities[i][s].weight, 0, M*N*sizeof(float));
                                                memset(datasets[d].fields[f].gridded_visibilities[i][s].S, 0, M*N*sizeof(int));
                                                memset(datasets[d].fields[f].gridded_visibilities[i][s].Vo, 0, M*N*sizeof(cufftComplex));
                                                memset(datasets[d].fields[f].gridded_visibilities[i][s].Vm, 0, M*N*sizeof(cufftComplex));

                                                //Allocate memory to backup original visibilities after gridding
                                                datasets[d].fields[f].backup_visibilities[i][s].uvw = (double3*)malloc(datasets[d].fields[f].numVisibilitiesPerFreqPerStoke[i][s]*sizeof(double3));
                                                datasets[d].fields[f].backup_visibilities[i][s].weight = (float*)malloc(datasets[d].fields[f].numVisibilitiesPerFreqPerStoke[i][s]*sizeof(float));
                                                datasets[d].fields[f].backup_visibilities[i][s].Vo = (cufftComplex*)malloc(datasets[d].fields[f].numVisibilitiesPerFreqPerStoke[i][s]*sizeof(cufftComplex));
                                                datasets[d].fields[f].backup_visibilities[i][s].Vm = (cufftComplex*)malloc(datasets[d].fields[f].numVisibilitiesPerFreqPerStoke[i][s]*sizeof(cufftComplex));

                                                memset(datasets[d].fields[f].backup_visibilities[i][s].uvw, 0, datasets[d].fields[f].numVisibilitiesPerFreqPerStoke[i][s]*sizeof(double3));
                                                memset(datasets[d].fields[f].backup_visibilities[i][s].weight, 0, datasets[d].fields[f].numVisibilitiesPerFreqPerStoke[i][s]*sizeof(float));
                                                memset(datasets[d].fields[f].backup_visibilities[i][s].Vo, 0, datasets[d].fields[f].numVisibilitiesPerFreqPerStoke[i][s]*sizeof(cufftComplex));
                                                memset(datasets[d].fields[f].backup_visibilities[i][s].Vm, 0, datasets[d].fields[f].numVisibilitiesPerFreqPerStoke[i][s]*sizeof(cufftComplex));

                                        }
                                }
                        }
                }
        }

        if(verbose_flag) {
                printf("Reading visibilities and FITS input files...\n");
        }

        for(int d=0; d<nMeasurementSets; d++) {
                if(apply_noise) {
                        iohandler->IoreadMS(datasets[d].name, datasets[d].fields, datasets[d].data, true, false, random_probability);
                }else{
                        iohandler->IoreadMS(datasets[d].name, datasets[d].fields, datasets[d].data, false, false, random_probability);
                }
        }

        this->visibilities = new Visibilities();
        this->visibilities->setMSDataset(datasets);
        this->visibilities->setNDatasets(nMeasurementSets);

        double deltax = RPDEG_D*DELTAX; //radians
        double deltay = RPDEG_D*DELTAY; //radians
        deltau = 1.0 / (M * deltax);
        deltav = 1.0 / (N * deltay);

        if(gridding) {
                printf("Doing gridding\n");
                omp_set_num_threads(gridding);
                for(int d=0; d<nMeasurementSets; d++)
                        do_gridding(datasets[d].fields, &datasets[d].data, deltau, deltav, M, N, robust_param);

                omp_set_num_threads(num_gpus);
        }
}

void MFS::setDevice()
{
        double deltax = RPDEG_D*DELTAX; //radians
        double deltay = RPDEG_D*DELTAY; //radians
        deltau = 1.0 / (M * deltax);
        deltav = 1.0 / (N * deltay);

        if(verbose_flag) {
                printf("MS File Successfully Read\n");
                if(beam_noise == -1) {
                        printf("Beam noise wasn't provided by the user... Calculating...\n");
                }
        }

        for(int d=0; d<nMeasurementSets; d++) {
                sum_weights = calculateNoise(datasets[d].fields, datasets[d].data, &total_visibilities, variables.blockSizeV, gridding);
        }

        this->visibilities->setTotalVisibilities(total_visibilities);

        for(int d=0; d<nMeasurementSets; d++) {
                for(int f=0; f<datasets[d].data.nfields; f++) {
                        cudaSetDevice(selected);
                        for(int i=0; i<datasets[d].data.total_frequencies; i++) {
                                for(int s=0; s<datasets[d].data.nstokes; s++) {
                                        gpuErrchk(cudaMalloc(&datasets[d].fields[f].device_visibilities[i][s].uvw,
                                                             sizeof(double3) * datasets[d].fields[f].numVisibilitiesPerFreqPerStoke[i][s]));
                                        gpuErrchk(cudaMalloc(&datasets[d].fields[f].device_visibilities[i][s].Vo,
                                                             sizeof(cufftComplex) * datasets[d].fields[f].numVisibilitiesPerFreqPerStoke[i][s]));
                                        gpuErrchk(cudaMalloc(&datasets[d].fields[f].device_visibilities[i][s].weight,
                                                             sizeof(float) * datasets[d].fields[f].numVisibilitiesPerFreqPerStoke[i][s]));
                                        gpuErrchk(cudaMalloc(&datasets[d].fields[f].device_visibilities[i][s].Vm,
                                                             sizeof(cufftComplex) * datasets[d].fields[f].numVisibilitiesPerFreqPerStoke[i][s]));
                                        gpuErrchk(cudaMalloc(&datasets[d].fields[f].device_visibilities[i][s].Vr,
                                                             sizeof(cufftComplex) * datasets[d].fields[f].numVisibilitiesPerFreqPerStoke[i][s]));
                                        gpuErrchk(cudaMemcpy(datasets[d].fields[f].device_visibilities[i][s].uvw, datasets[d].fields[f].visibilities[i][s].uvw,
                                                             sizeof(double3) * datasets[d].fields[f].numVisibilitiesPerFreqPerStoke[i][s],
                                                             cudaMemcpyHostToDevice));

                                        gpuErrchk(cudaMemcpy(datasets[d].fields[f].device_visibilities[i][s].weight,
                                                             datasets[d].fields[f].visibilities[i][s].weight,
                                                             sizeof(float) * datasets[d].fields[f].numVisibilitiesPerFreqPerStoke[i][s],
                                                             cudaMemcpyHostToDevice));

                                        gpuErrchk(cudaMemcpy(datasets[d].fields[f].device_visibilities[i][s].Vo, datasets[d].fields[f].visibilities[i][s].Vo,
                                                             sizeof(cufftComplex) * datasets[d].fields[f].numVisibilitiesPerFreqPerStoke[i][s],
                                                             cudaMemcpyHostToDevice));

                                        gpuErrchk(cudaMemset(datasets[d].fields[f].device_visibilities[i][s].Vr, 0,
                                                             sizeof(cufftComplex) * datasets[d].fields[f].numVisibilitiesPerFreqPerStoke[i][s]));
                                        gpuErrchk(cudaMemset(datasets[d].fields[f].device_visibilities[i][s].Vm, 0,
                                                             sizeof(cufftComplex) * datasets[d].fields[f].numVisibilitiesPerFreqPerStoke[i][s]));
                                }
                        }

                        gpuErrchk(cudaMalloc((void**)&datasets[d].fields[f].atten_image, sizeof(float)*M*N));
                        gpuErrchk(cudaMemset(datasets[d].fields[f].atten_image, 0, sizeof(float)*M*N));

                }
        }


        max_number_vis = 0;
        for(int d=0; d<nMeasurementSets; d++) {
                if(datasets[d].data.max_number_visibilities_in_channel_and_stokes > max_number_vis)
                        max_number_vis = datasets[d].data.max_number_visibilities_in_channel_and_stokes;
        }

        if(max_number_vis == 0) {
                printf("Max number of visibilities cannot be zero for image synthesis\n");
                exit(-1);
        }

        this->visibilities->setMaxNumberVis(max_number_vis);

        for(int g=0; g<num_gpus; g++) {
                cudaSetDevice((g%num_gpus) + firstgpu);
                gpuErrchk(cudaMalloc((void**)&vars_gpu[g].device_dchi2, sizeof(float)*M*N));
                gpuErrchk(cudaMemset(vars_gpu[g].device_dchi2, 0, sizeof(float)*M*N));

                gpuErrchk(cudaMalloc(&vars_gpu[g].device_chi2, sizeof(float)*max_number_vis));
                gpuErrchk(cudaMemset(vars_gpu[g].device_chi2, 0, sizeof(float)*max_number_vis));
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
                printf("FITS: Center pix: (%lf,%lf)\n", crpix1-1, crpix2-1);
        }

        double lobs, mobs, lphs, mphs;
        double dcosines_l_pix_ref, dcosines_m_pix_ref, dcosines_l_pix_phs, dcosines_m_pix_phs;
        for(int d=0; d<nMeasurementSets; d++) {
                if(verbose_flag)
                        printf("Dataset: %s\n", datasets[d].name);
                for(int f=0; f<datasets[d].data.nfields; f++) {

                        direccos(datasets[d].fields[f].ref_ra, datasets[d].fields[f].ref_dec, raimage, decimage, &lobs,  &mobs);
                        direccos(datasets[d].fields[f].phs_ra, datasets[d].fields[f].phs_dec, raimage, decimage, &lphs,  &mphs);

                        dcosines_l_pix_ref = lobs/ -deltax; // Radians to pixels
                        dcosines_m_pix_ref = mobs/fabs(deltay); // Radians to pixels

                        dcosines_l_pix_phs = lphs/ -deltax; // Radians to pixels
                        dcosines_m_pix_phs = mphs/fabs(deltay); // Radians to pixels

                        if(verbose_flag)
                        {
                                printf("Ref: l (pix): %e, m (pix): %e\n", dcosines_l_pix_ref, dcosines_m_pix_ref);
                                printf("Phase: l (pix): %e, m (pix): %e\n", dcosines_l_pix_phs, dcosines_m_pix_phs);

                        }


                        datasets[d].fields[f].ref_xobs = (crpix1 - 1.0f) + dcosines_l_pix_ref;// + 6.0f;
                        datasets[d].fields[f].ref_yobs = (crpix2 - 1.0f) + dcosines_m_pix_ref;// - 7.0f;

                        datasets[d].fields[f].phs_xobs = (crpix1 - 1.0f) + dcosines_l_pix_phs;// + 5.0f;
                        datasets[d].fields[f].phs_yobs = (crpix2 - 1.0f) + dcosines_m_pix_phs;// - 7.0f;


                        if(verbose_flag) {
                                printf("Ref: Field %d - Ra: %.16e (rad), dec: %.16e (rad), x0: %f (pix), y0: %f (pix)\n", f, datasets[d].fields[f].ref_ra, datasets[d].fields[f].ref_dec,
                                       datasets[d].fields[f].ref_xobs, datasets[d].fields[f].ref_yobs);
                                printf("Phase: Field %d - Ra: %.16e (rad), dec: %.16e (rad), x0: %f (pix), y0: %f (pix)\n", f, datasets[d].fields[f].phs_ra, datasets[d].fields[f].phs_dec,
                                       datasets[d].fields[f].phs_xobs, datasets[d].fields[f].phs_yobs);
                        }

                        if(datasets[d].fields[f].ref_xobs < 0 || datasets[d].fields[f].ref_xobs >= M || datasets[d].fields[f].ref_xobs < 0 || datasets[d].fields[f].ref_yobs >= N) {
                                printf("Dataset: %s\n", datasets[d].name);
                                printf("Pointing reference center (%f,%f) is outside the range of the image\n", datasets[d].fields[f].ref_xobs, datasets[d].fields[f].ref_yobs);
                                goToError();
                        }

                        if(datasets[d].fields[f].phs_xobs < 0 || datasets[d].fields[f].phs_xobs >= M || datasets[d].fields[f].phs_xobs < 0 || datasets[d].fields[f].phs_yobs >= N) {
                                printf("Dataset: %s\n", datasets[d].name);
                                printf("Pointing phase center (%f,%f) is outside the range of the image\n", datasets[d].fields[f].phs_xobs, datasets[d].fields[f].phs_yobs);
                                goToError();
                        }
                }
        }
        ////////////////////////////////////////////////////////MAKE STARTING IMAGE////////////////////////////////////////////////////////

        host_I = (float*)malloc(M*N*sizeof(float)*image_count);

        for(int i=0; i<M; i++) {
                for(int j=0; j<N; j++) {
                        for(int k=0; k<image_count; k++) {
                                host_I[N*M*k+N*i+j] = initial_values[k];
                        }
                }
        }

        ////////////////////////////////////////////////CUDA MEMORY ALLOCATION FOR DEVICE///////////////////////////////////////////////////

        for(int g=0; g<num_gpus; g++) {
                cudaSetDevice((g%num_gpus) + firstgpu);
                gpuErrchk(cudaMalloc((void**)&vars_gpu[g].device_V, sizeof(cufftComplex)*M*N*image_count));
                gpuErrchk(cudaMalloc((void**)&vars_gpu[g].device_IQUV_nu, sizeof(cufftComplex)*M*N*image_count));
        }


        cudaSetDevice(firstgpu);

        gpuErrchk(cudaMalloc((void**)&device_Image, sizeof(float)*M*N*image_count));
        gpuErrchk(cudaMemset(device_Image, 0, sizeof(float)*M*N*image_count));

        gpuErrchk(cudaMemcpy(device_Image, host_I, sizeof(float)*N*M*image_count, cudaMemcpyHostToDevice));

        gpuErrchk(cudaMalloc((void**)&device_noise_image, sizeof(float)*M*N));
        gpuErrchk(cudaMemset(device_noise_image, 0, sizeof(float)*M*N));

        gpuErrchk(cudaMalloc((void**)&device_weight_image, sizeof(float)*M*N));
        gpuErrchk(cudaMemset(device_weight_image, 0, sizeof(float)*M*N));



        for(int g=0; g<num_gpus; g++) {
                cudaSetDevice((g%num_gpus) + firstgpu);
                gpuErrchk(cudaMemset(vars_gpu[g].device_V, 0, sizeof(cufftComplex)*M*N*image_count));
                gpuErrchk(cudaMemset(vars_gpu[g].device_I_nu, 0, sizeof(cufftComplex)*M*N*image_count));

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


        initFFT(vars_gpu, M, N, firstgpu, num_gpus);

        //Time is taken from first kernel
        t = clock();
        start = omp_get_wtime();
        for(int d=0; d<nMeasurementSets; d++) {
                for(int f=0; f < datasets[d].data.nfields; f++) {
                        cudaSetDevice(selected);
                        for(int i=0; i<datasets[d].data.total_frequencies; i++) {
                                for(int s=0; s<datasets[d].data.nstokes; s++) {
                                        hermitianSymmetry << < datasets[d].fields[f].visibilities[i][s].numBlocksUV,
                                                datasets[d].fields[f].visibilities[i][s].threadsPerBlockUV >> >
                                        (datasets[d].fields[f].device_visibilities[i][s].uvw, datasets[d].fields[f].device_visibilities[i][s].Vo, datasets[d].fields[f].nu[i], datasets[d].fields[f].numVisibilitiesPerFreqPerStoke[i][s]);
                                        gpuErrchk(cudaDeviceSynchronize());
                                }
                        }
                }

                cudaSetDevice(selected);
                for(int f=0; f<datasets[d].data.nfields; f++) {
                        for(int i=0; i<datasets[d].data.total_frequencies; i++) {
                                if(datasets[d].fields[f].numVisibilitiesPerFreq[i] > 0) {
                                        total_attenuation<<<numBlocksNN, threadsPerBlockNN>>>(datasets[d].fields[f].atten_image, datasets[d].antenna_diameter, datasets[d].pb_factor, datasets[d].pb_cutoff, datasets[d].fields[f].nu[i], datasets[d].fields[f].ref_xobs, datasets[d].fields[f].ref_yobs, DELTAX, DELTAY, N);
                                        gpuErrchk(cudaDeviceSynchronize());
                                }
                        }
                }


                for(int f=0; f<datasets[d].data.nfields; f++) {
                        if(datasets[d].fields[f].valid_frequencies > 0) {
                                if(num_gpus == 1) {
                                        cudaSetDevice(selected);
                                        mean_attenuation<<<numBlocksNN, threadsPerBlockNN>>>(datasets[d].fields[f].atten_image, datasets[d].fields[f].valid_frequencies, N);
                                        gpuErrchk(cudaDeviceSynchronize());
                                }else{
                                        cudaSetDevice(firstgpu);
                                        mean_attenuation<<<numBlocksNN, threadsPerBlockNN>>>(datasets[d].fields[f].atten_image, datasets[d].fields[f].valid_frequencies, N);
                                        gpuErrchk(cudaDeviceSynchronize());
                                }
                                if(print_images) {
                                        std::string atten_name =  "dataset_" + std::to_string(d) + "_atten";
                                        iohandler->IoPrintImageIteration(datasets[d].fields[f].atten_image, mod_in, mempath, atten_name.c_str(), "", f, 0, 1.0, M, N);
                                }
                        }
                }
        }



        if(num_gpus == 1) {
                cudaSetDevice(selected);
        }else{
                cudaSetDevice(firstgpu);
        }

        for(int d=0; d<nMeasurementSets; d++) {
                for(int f=0; f<datasets[d].data.nfields; f++) {
                        weight_image<<<numBlocksNN, threadsPerBlockNN>>>(device_weight_image, datasets[d].fields[f].atten_image, noise_jypix, N);
                        gpuErrchk(cudaDeviceSynchronize());
                }
        }

        noise_image<<<numBlocksNN, threadsPerBlockNN>>>(device_noise_image, device_weight_image, noise_jypix, N);
        gpuErrchk(cudaDeviceSynchronize());
        if(print_images)
                iohandler->IoPrintImage(device_noise_image, mod_in, mempath, "noise.fits", "", 0, 0, 1.0, M, N);


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
        for(int d=0; d<nMeasurementSets; d++) {
                for(int f=0; f<datasets[d].data.nfields; f++) {
                        cudaFree(datasets[d].fields[f].atten_image);
                }
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
                iohandler->IoPrintImage(image->getImage(), mod_in, "", "alpha.fits", "", iter, 1, 1.0, M, N);
        }else{
                (IoOrderEnd)(image->getImage(), iohandler);
        }

        if(print_errors) /* flag for print error image */
        {
                if(this->error == NULL)
                {
                        this->error = Singleton<ErrorFactory>::Instance().CreateError(0);
                }
                /* code to calculate error */
                /* make void * params */
                printf("Calculating Error Images\n");
                this->error->calculateErrorImage(this->image, this->visibilities);
                if(IoOrderError == NULL) {
                        iohandler->IoPrintImage(image->getErrorImage(), mod_in, "", "error_Inu_0.fits", "JY/PIXEL", iter, 0, 1.0, M, N);
                        iohandler->IoPrintImage(image->getErrorImage(), mod_in, "", "error_alpha.fits", "", iter, 1, 1.0, M, N);
                }else{
                        (IoOrderError)(image->getErrorImage(), iohandler);
                }

        }

        if(!gridding)
        {
                //Saving residuals to disk
                for(int d=0; d<nMeasurementSets; d++) {
                        residualsToHost(datasets[d].fields, datasets[d].data, num_gpus, firstgpu);
                }
        }else{
                double deltax = RPDEG_D*DELTAX; //radians
                double deltay = RPDEG_D*DELTAY; //radians
                deltau = 1.0 / (M * deltax);
                deltav = 1.0 / (N * deltay);

                printf("Visibilities are gridded, we will need to de-grid to save them in a Measurement Set File\n");
                omp_set_num_threads(gridding);
                for(int d=0; d<nMeasurementSets; d++)
                        degridding(datasets[d].fields, datasets[d].data, deltau, deltav, num_gpus, firstgpu, variables.blockSizeV, M, N);

                omp_set_num_threads(num_gpus);

                for(int d=0; d<nMeasurementSets; d++)
                        residualsToHost(datasets[d].fields, datasets[d].data, num_gpus, firstgpu);

        }

        printf("Saving residuals to MS...\n");
        for(int d=0; d<nMeasurementSets; d++)
                iohandler->IowriteMS(datasets[d].name, datasets[d].oname, datasets[d].fields, datasets[d].data, random_probability, false, false, false, verbose_flag);

        printf("Residuals saved.\n");


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

        for(int d=0; d<nMeasurementSets; d++) {
                for(int f=0; f<datasets[d].data.nfields; f++) {
                        for(int i=0; i<datasets[d].data.total_frequencies; i++) {

                                if(num_gpus > 1) {
                                        cudaSetDevice((i%num_gpus) + firstgpu);
                                }
                                for(int s=0; s<datasets[d].data.nstokes; s++) {
                                        cudaFree(datasets[d].fields[f].device_visibilities[i][s].uvw);
                                        cudaFree(datasets[d].fields[f].device_visibilities[i][s].weight);
                                        cudaFree(datasets[d].fields[f].device_visibilities[i][s].Vr);
                                        cudaFree(datasets[d].fields[f].device_visibilities[i][s].Vm);
                                        cudaFree(datasets[d].fields[f].device_visibilities[i][s].Vo);
                                }

                        }
                }
        }

        printf("Freeing cuFFT plans\n");
        for(int g=0; g<num_gpus; g++) {
                cudaSetDevice((g%num_gpus) + firstgpu);
                cufftDestroy(vars_gpu[g].plan);
        }

        printf("Freeing host memory\n");
        for(int d=0; d<nMeasurementSets; d++) {
                for(int f=0; f<datasets[d].data.nfields; f++) {
                        for(int i=0; i<datasets[d].data.total_frequencies; i++) {
                                for(int s=0; s<datasets[d].data.nstokes; s++) {
                                        if (datasets[d].fields[f].numVisibilitiesPerFreqPerStoke[i][s] > 0) {
                                                free(datasets[d].fields[f].visibilities[i][s].uvw);
                                                free(datasets[d].fields[f].visibilities[i][s].weight);
                                                free(datasets[d].fields[f].visibilities[i][s].Vo);
                                                free(datasets[d].fields[f].visibilities[i][s].Vm);
                                        }
                                }
                        }
                }
        }



        cudaFree(device_Image);

        for(int g=0; g<num_gpus; g++) {
                cudaSetDevice((g%num_gpus) + firstgpu);
                cudaFree(vars_gpu[g].device_V);
                cudaFree(vars_gpu[g].device_I_nu);
        }


        cudaSetDevice(firstgpu);


        cudaFree(device_noise_image);

        cudaFree(device_dphi);
        cudaFree(device_dchi2_total);
        cudaFree(device_dS);

        cudaFree(device_S);

        free(host_I);
        free(msinput);
        free(t_telescope);
        free(msoutput);
        free(modinput);

        for(int i=0; i< nMeasurementSets; i++) {
                free(datasets[i].name);
                free(datasets[i].oname);
        }

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
