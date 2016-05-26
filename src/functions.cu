#include "functions.cuh"


extern long M;
extern long N;
extern int numVisibilities;
extern int iterations;
extern int iter;


extern cufftHandle plan1GPU;
extern cufftComplex *device_I;
extern cufftComplex *device_V;
extern cufftComplex *device_noise_image;
extern cufftComplex *device_fg_image;
extern cufftComplex *device_image;

extern float *device_dphi;


extern float *device_chi2;
extern float *device_H;
extern float *device_dchi2_total;
extern float *device_dH;

extern dim3 threadsPerBlockNN;
extern dim3 numBlocksNN;

extern int threadsVectorNN;
extern int blocksVectorNN;

extern float difmap_noise;
extern float fg_scale;

extern float global_xobs;
extern float global_yobs;

extern float DELTAX;
extern float DELTAY;
extern float deltau;
extern float deltav;

extern float noise_cut;
extern float MINPIX;
extern float minpix_factor;
extern float lambda;
extern float ftol;
extern float random_probability;
extern int positivity;


extern float beam_noise;
extern float beam_bmaj;
extern float beam_bmin;
extern double ra;
extern double dec;
extern double obsra;
extern double obsdec;
extern int crpix1;
extern int crpix2;
extern freqData data;
extern VPF *device_vars;
extern Vis *visibilities;
extern Vis *device_visibilities;

extern int num_gpus;
extern int selected;

extern char* mempath;

extern fitsfile *mod_in;
extern int status_mod_in;
extern int verbose_flag;

__host__ void goToError()
{
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

  printf("An error has ocurred, exiting\n");
  exit(0);

}

__host__ freqData getFreqs(char * file)
{
   freqData freqsAndVisibilities;
   sqlite3 *db;
   sqlite3_stmt *stmt;
   char *err_msg = 0;

   int rc = sqlite3_open(file, &db);

   if (rc != SQLITE_OK) {
    printf("Cannot open database: %s\n", sqlite3_errmsg(db));
    sqlite3_close(db);
    goToError();
  }else{
    if(verbose_flag){
      printf("Database connection okay!\n");
    }
  }

  char *sql = "SELECT n_internal_frequencies as nfreq FROM header";
  rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL );
  sqlite3_step(stmt);
  freqsAndVisibilities.n_internal_frequencies = sqlite3_column_int(stmt, 0);

  freqsAndVisibilities.channels = (int*)malloc(freqsAndVisibilities.n_internal_frequencies*sizeof(int));

  sql = "SELECT COUNT(*) as channels FROM channels WHERE internal_freq_id = ?";
  rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL );
  if (rc != SQLITE_OK ) {
    printf("Cannot open database: %s\n", sqlite3_errmsg(db));
    sqlite3_close(db);
    goToError();
  }

  for(int i=0; i< freqsAndVisibilities.n_internal_frequencies; i++){
      sqlite3_bind_int(stmt, 1, i);
      sqlite3_step(stmt);
      freqsAndVisibilities.channels[i] = sqlite3_column_int(stmt, 0);
      sqlite3_reset(stmt);
  }

  int total_frequencies = 0;
  for(int i=0; i <freqsAndVisibilities.n_internal_frequencies; i++){
    for(int j=0; j < freqsAndVisibilities.channels[i]; j++){
      total_frequencies++;
    }
  }

  freqsAndVisibilities.total_frequencies = total_frequencies;
  freqsAndVisibilities.numVisibilitiesPerFreq = (long*)malloc(freqsAndVisibilities.total_frequencies*sizeof(long));


  sql = "SELECT COUNT(*) AS visibilitiesPerFreq FROM(SELECT samples.u as u, samples.v as v, samples.w as w, visibilities.stokes as stokes, samples.id_field as id_field, visibilities.Re as Re, visibilities.Im as Im, weights.weight as We, id_antenna1, id_antenna2, channels.internal_freq_id as  internal_frequency_id , visibilities.channel as channel, frequency FROM visibilities,samples,weights,channels WHERE visibilities.flag=0 and samples.flag_row=0 and visibilities.id_sample = samples.id and weights.id_sample=samples.id and weights.stokes=visibilities.stokes and channels.internal_freq_id=samples.internal_freq_id and visibilities.channel=channels.channel) WHERE internal_frequency_id = ? AND channel = ?";
  rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL );
  if (rc != SQLITE_OK ) {
    printf("Cannot open database: %s\n", sqlite3_errmsg(db));
    sqlite3_close(db);
    goToError();
  }
  int h=0;
  for(int i=0; i <freqsAndVisibilities.n_internal_frequencies; i++){
    for(int j=0; j < freqsAndVisibilities.channels[i]; j++){
      sqlite3_bind_int(stmt, 1, i);
      sqlite3_bind_int(stmt, 2, j);
      sqlite3_step(stmt);
      freqsAndVisibilities.numVisibilitiesPerFreq[h] = (long)sqlite3_column_int(stmt, 0);
      sqlite3_reset(stmt);
      sqlite3_clear_bindings(stmt);
      h++;
    }
  }

  sqlite3_finalize(stmt);
  sqlite3_close(db);
  return freqsAndVisibilities;
}

__host__ long NearestPowerOf2(long n)
{
  if (!n) return n;  //(0 == 2^0)

  int x = 1;
  while(x < n)
  {
      x <<= 1;
  }
  return x;
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
  if((fp = fopen(file, "r")) == NULL){
    printf("ERROR. The input file wasn't provided by the user.\n");
    goToError();
  }else{
    while(true){
      int ret = fscanf(fp, "%s %e", item, &status);

      if(ret==EOF){
        break;
      }else{
        if (strcmp(item,"lambda_entropy")==0) {
          lambda = status;
        }else if (strcmp(item,"noise_cut")==0){
          noise_cut = status;
        }else if(strcmp(item,"minpix_factor")==0){
          minpix_factor = status;
        } else if(strcmp(item,"ftol")==0){
          ftol = status;
        } else if(strcmp(item,"random_probability")==0){
          random_probability = status;
        } else if(strcmp(item,"positivity")==0){
            positivity = status;
        }else{
          break;
        }
      }
    }
  }
}
__host__ void readMS(char *file, char *file2, char *file3, Vis *visibilities) {
  ///////////////////////////////////////////////////FITS READING///////////////////////////////////////////////////////////
  status_mod_in = 0;
  fits_open_file(&mod_in, file2, 0, &status_mod_in);
  if (status_mod_in) {
    fits_report_error(stderr, status_mod_in); /* print error message */
    goToError();
  }


  fits_read_key(mod_in, TFLOAT, "CDELT1", &DELTAX, NULL, &status_mod_in);
  fits_read_key(mod_in, TFLOAT, "CDELT2", &DELTAY, NULL, &status_mod_in);
  fits_read_key(mod_in, TDOUBLE, "CRVAL1", &ra, NULL, &status_mod_in);
  fits_read_key(mod_in, TDOUBLE, "CRVAL2", &dec, NULL, &status_mod_in);
  fits_read_key(mod_in, TINT, "CRPIX1", &crpix1, NULL, &status_mod_in);
  fits_read_key(mod_in, TINT, "CRPIX2", &crpix2, NULL, &status_mod_in);
  fits_read_key(mod_in, TLONG, "NAXIS1", &M, NULL, &status_mod_in);
  fits_read_key(mod_in, TLONG, "NAXIS2", &N, NULL, &status_mod_in);
  if (status_mod_in) {
    fits_report_error(stderr, status_mod_in); /* print error message */
    goToError();
  }


  fitsfile *fpointer2;
  int status2 = 0;
  fits_open_file(&fpointer2, file3, 0, &status2);
  if (status2) {
    fits_report_error(stderr, status2); /* print error message */
    goToError();
  }
  fits_read_key(fpointer2, TFLOAT, "NOISE", &beam_noise, NULL, &status2);
  fits_read_key(fpointer2, TFLOAT, "BMAJ", &beam_bmaj, NULL, &status2);
  fits_read_key(fpointer2, TFLOAT, "BMIN", &beam_bmin, NULL, &status2);
  if (status2) {
    fits_report_error(stderr, status2); /* print error message */
    goToError();
  }
  fits_close_file(fpointer2, &status2);
  if (status2) {
    fits_report_error(stderr, status2); /* print error message */
    goToError();
  }
  if(verbose_flag){
    printf("FITS Files READ\n");
  }

  ///////////////////////////////////////////////////MS SQLITE READING/////////////////////////////////////////////////////////
  sqlite3 *db;
  sqlite3_stmt *stmt;
  char *error = 0;
  int k = 0;
  int h = 0;


  int rc = sqlite3_open(file, &db);
  if (rc != SQLITE_OK) {
   printf("Cannot open database: %s\n", sqlite3_errmsg(db));
   sqlite3_close(db);
   goToError();
  }else{
   if(verbose_flag){
     printf("Database connection okay again!\n");
   }
 }

  char *sql;
  sql = "SELECT id, stokes, u, v, Re, Im, We FROM(SELECT samples.id as id, samples.u as u, samples.v as v, samples.w as w, visibilities.stokes as stokes, samples.id_field as id_field, visibilities.Re as Re, visibilities.Im as Im, weights.weight as We, id_antenna1, id_antenna2, channels.internal_freq_id as  internal_frequency_id , visibilities.channel as channel, frequency FROM visibilities,samples,weights,channels WHERE visibilities.flag=0 and samples.flag_row=0 and visibilities.id_sample = samples.id and weights.id_sample=samples.id and weights.stokes=visibilities.stokes and channels.internal_freq_id=samples.internal_freq_id and visibilities.channel=channels.channel) WHERE internal_frequency_id = ? AND channel = ?";
  rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
  if (rc != SQLITE_OK ) {
    printf("Cannot execute SELECT: %s\n", sqlite3_errmsg(db));
    sqlite3_close(db);
    goToError();
  }

  if(random_probability!=0.0){
    float u;
    SelectStream(0);
    PutSeed(1);
    for(int i=0; i < data.n_internal_frequencies; i++){
      for(int j=0; j < data.channels[i]; j++){
        sqlite3_bind_int(stmt, 1, i);
        sqlite3_bind_int(stmt, 2, j);
        while (1) {
          int s = sqlite3_step(stmt);
          if (s == SQLITE_ROW) {
            u = Random();
            if(u<1-random_probability){
              visibilities[k].id[h] = sqlite3_column_int(stmt, 0);
              visibilities[k].stokes[h] = sqlite3_column_int(stmt, 1);
              visibilities[k].u[h] = sqlite3_column_double(stmt, 2);
              visibilities[k].v[h] = sqlite3_column_double(stmt, 3);
              visibilities[k].Vo[h].x = sqlite3_column_double(stmt, 4);
              visibilities[k].Vo[h].y = sqlite3_column_double(stmt, 5);
              visibilities[k].weight[h] = sqlite3_column_double(stmt, 6);
              h++;
            }
          }else if(s == SQLITE_DONE) {
            break;
          }else{
            printf("Database queries failed ERROR: %d.\n", s);
            goToError();
          }
        }
        data.numVisibilitiesPerFreq[k] = (h+1);
        realloc(visibilities[k].id, (h+1)*sizeof(int));
        realloc(visibilities[k].stokes, (h+1)*sizeof(int));
        realloc(visibilities[k].u, (h+1)*sizeof(float));
        realloc(visibilities[k].v, (h+1)*sizeof(float));
        realloc(visibilities[k].Vo, (h+1)*sizeof(cufftComplex));
        realloc(visibilities[k].weight, (h+1)*sizeof(float));
        h=0;
        sqlite3_reset(stmt);
        sqlite3_clear_bindings(stmt);
        k++;
      }
    }
  }else{
    for(int i=0; i < data.n_internal_frequencies; i++){
      for(int j=0; j < data.channels[i]; j++){
        sqlite3_bind_int(stmt, 1, i);
        sqlite3_bind_int(stmt, 2, j);
        while (1) {
          int s = sqlite3_step(stmt);
          if (s == SQLITE_ROW) {
            visibilities[k].id[h] = sqlite3_column_int(stmt, 0);
            visibilities[k].stokes[h] = sqlite3_column_int(stmt, 1);
            visibilities[k].u[h] = sqlite3_column_double(stmt, 2);
            visibilities[k].v[h] = sqlite3_column_double(stmt, 3);
            visibilities[k].Vo[h].x = sqlite3_column_double(stmt, 4);
            visibilities[k].Vo[h].y = sqlite3_column_double(stmt, 5);
            visibilities[k].weight[h] = sqlite3_column_double(stmt, 6);
            h++;
          }else if(s == SQLITE_DONE) {
            break;
          }else{
            printf("Database queries failed ERROR: %d.\n", s);
            goToError();
          }
        }
        h=0;
        sqlite3_reset(stmt);
        sqlite3_clear_bindings(stmt);
        k++;
      }
    }
  }
  if(verbose_flag){
    printf("Visibilities read!\n");
  }



  sql = "SELECT ra, dec FROM fields";
  rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
  if (rc != SQLITE_OK ) {
    printf("SQL error: %s\n", error);
    sqlite3_free(error);
    sqlite3_close(db);
    goToError();
  }
  sqlite3_step(stmt);
  obsra = sqlite3_column_double(stmt, 0);
  obsdec = sqlite3_column_double(stmt, 1);
  if(verbose_flag){
    printf("Center read!\n");
  }

  sql = "SELECT frequency as freq_vector FROM channels WHERE internal_freq_id = ? AND channel = ?";
  rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL );
  if (rc != SQLITE_OK ) {
    printf("SQL error: %s\n", error);
    sqlite3_free(error);
    sqlite3_close(db);
    goToError();
  }
  h=0;
  for(int i = 0; i < data.n_internal_frequencies; i++){
    for(int j = 0; j < data.channels[i]; j++){
      sqlite3_bind_int(stmt, 1, i);
      sqlite3_bind_int(stmt, 2, j);
      while (1) {
        int s = sqlite3_step(stmt);
        if (s == SQLITE_ROW) {
          visibilities[h].freq = sqlite3_column_double(stmt, 0);
          h++;
        }else if(s == SQLITE_DONE) {
          break;
        }else{
          printf("Database queries failed ERROR %d at %d and %d.\n",s, i, j);
          goToError();
        }
      }
      sqlite3_reset(stmt);
      sqlite3_clear_bindings(stmt);
    }
  }
  if(verbose_flag){
    printf("Frequencies read!\n");
  }

  sqlite3_close(db);
}


__host__ void residualsToHost(Vis *device_visibilities, Vis *visibilities, freqData data){
  printf("Saving residuals to host memory\n");
  if(num_gpus == 1){
    for(int i=0; i<data.total_frequencies; i++){
      gpuErrchk(cudaMemcpy(visibilities[i].Vr, device_visibilities[i].Vr, sizeof(cufftComplex)*data.numVisibilitiesPerFreq[i], cudaMemcpyDeviceToHost));
    }
  }else{
    for(int i=0; i<data.total_frequencies; i++){
      cudaSetDevice(i%num_gpus);
      gpuErrchk(cudaMemcpy(visibilities[i].Vr, device_visibilities[i].Vr, sizeof(cufftComplex)*data.numVisibilitiesPerFreq[i], cudaMemcpyDeviceToHost));
    }
  }

  for(int i=0; i<data.total_frequencies; i++){
    for(int j=0; j<data.numVisibilitiesPerFreq[i];j++){
      if(visibilities[i].u[j]<0){
        visibilities[i].Vr[j].y *= -1;
      }
    }
  }

}

__host__ void writeMS(char *file, Vis *visibilities) {
  sqlite3 *db;
  sqlite3_stmt *stmt;
  int rc;

  rc = sqlite3_open(file, &db);
  if (rc != SQLITE_OK) {
   printf("Cannot open output database: %s\n", sqlite3_errmsg(db));
   sqlite3_close(db);
   goToError();
  }else{
   printf("Output Database connection okay!\n");
 }


 char *sql = "DELETE FROM visibilities WHERE flag = 0";
 //rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
 rc = sqlite3_exec(db, sql, NULL, NULL, 0);
 if (rc != SQLITE_OK ) {
   printf("Cannot execute DELETE: %s\n", sqlite3_errmsg(db));
   sqlite3_close(db);
   goToError();
 }else{
   printf("Observed visibilities deleted from output file!\n");
 }
 sqlite3_exec(db, "PRAGMA synchronous = OFF", NULL, NULL, 0);
 sqlite3_exec(db, "PRAGMA journal_mode = MEMORY", NULL, NULL, 0);

 int l = 0;
 sql = "INSERT INTO visibilities (id_sample, stokes, channel, re, im, flag) VALUES (?, ?, ?, ?, ?, 0)";
 rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL );
 if (rc != SQLITE_OK ) {
   printf("Cannot execute INSERT: %s\n", sqlite3_errmsg(db));
   sqlite3_close(db);
   goToError();
 }else{
   printf("Saving residuals to file. Please wait...\n");
 }

 for(int i=0; i<data.n_internal_frequencies; i++){
   for(int j=0; j<data.channels[i]; j++){
     for(int k=0; k<data.numVisibilitiesPerFreq[l]; k++){
       sqlite3_bind_int(stmt, 1, visibilities[l].id[k]);
       sqlite3_bind_int(stmt, 2, visibilities[l].stokes[k]);
       sqlite3_bind_int(stmt, 3, j);
       sqlite3_bind_double(stmt, 4, -visibilities[l].Vr[k].x);
       sqlite3_bind_double(stmt, 5, -visibilities[l].Vr[k].y);
       int s = sqlite3_step(stmt);
       sqlite3_clear_bindings(stmt);
       sqlite3_reset(stmt);
       printf("\rSPW[%d/%d], channel[%d/%d]: %d/%d ====> %d %%", i+1, data.n_internal_frequencies, j+1, data.channels[i], k+1, data.numVisibilitiesPerFreq[l], int((k*100.0)/data.numVisibilitiesPerFreq[l])+1);
       fflush(stdout);
     }
     printf("\n");
     l++;
   }
 }
 sqlite3_finalize(stmt);


 l=0;

 sql = "UPDATE samples SET u = ?, v = ? WHERE id = ? AND flag_row = 0 AND internal_freq_id = ?";

 rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL );
 if (rc != SQLITE_OK ) {
   printf("Cannot execute UPDATE: %s\n", sqlite3_errmsg(db));
   sqlite3_close(db);
   goToError();
 }else{
   printf("Passing u,v to file. Please wait...\n");
 }

 for(int i=0; i<data.n_internal_frequencies; i++){
    for(int j=0; j<data.channels[i]; j++){
      for(int k=0; k<data.numVisibilitiesPerFreq[l]; k++){
        sqlite3_bind_double(stmt, 1, visibilities[l].u[k]);
        sqlite3_bind_double(stmt, 2, visibilities[l].v[k]);
        sqlite3_bind_int(stmt, 3, visibilities[l].id[k]);
        sqlite3_bind_int(stmt, 4, i);
        int s = sqlite3_step(stmt);
        sqlite3_clear_bindings(stmt);
        sqlite3_reset(stmt);
        //printf("u: %f, v: %f, id: %d, freq: %d", visibilities[l].u[k], visibilities[l].v[k], visibilities[l].id[k], i);
        printf("\rSPW[%d/%d], channel[%d/%d]: %d/%d ====> %d %%", i+1, data.n_internal_frequencies, j+1, data.channels[i], k+1, data.numVisibilitiesPerFreq[l], int((k*100.0)/data.numVisibilitiesPerFreq[l])+1);
        fflush(stdout);
      }
      printf("\n");
      l++;
    }
  }


 sqlite3_finalize(stmt);
 sqlite3_close(db);

}

__host__ float2 minmaxV(cufftComplex *visibilities, int N){
  float max = visibilities[0].x;
  float min = visibilities[0].x;
  for(int i = 0; i<N; i++)
  {
        if(visibilities[i].x > max){
          max = visibilities[i].x;
        }

        if(visibilities[i].x < min){
          min = visibilities[i].x;
        }

  }
  float2 result;
  result.x = min;
  result.y = max;
  return result;
}


__host__ void print_help() {
	printf("Example: ./bin/gpuvmem options [ arguments ...]\n");
	printf("    -h  --help       Shows this\n");
  printf(	"    -X  --blockSizeX      Block X Size for Image (Needs to be pow of 2)\n");
  printf(	"    -Y  --blockSizeY      Block Y Size for Image (Needs to be pow of 2)\n");
  printf(	"    -V  --blockSizeV      Block Size for Visibilities (Needs to be pow of 2)\n");
  printf(	"    -i  --input      The name of the input file of visibilities(SQLite)\n");
  printf(	"    -o  --output     The name of the output file of residual visibilities(SQLite)\n");
  printf("    -I  --inputdat   The name of the input file of parameters\n");
  printf("    -m  --modin      mod_in_0 FITS file\n");
  printf("    -b  --beam       beam_0 FITS file\n");
  printf("    -p  --path       MEM folder path to save FITS images. With last / included. (Example ./../mem/)\n");
  printf("    -M  --multigpu   Number of GPUs to use multiGPU image synthesis (Default OFF => 0)\n");
  printf("    -s  --select     If multigpu option is OFF, then select the GPU ID of the GPU you will work on. (Default = 0)\n");
}

__host__ Vars getOptions(int argc, char **argv) {
	Vars variables;
	variables.input = (char*) malloc(2000*sizeof(char));
  variables.output = (char*) malloc(2000*sizeof(char));
  variables.inputdat = (char*) malloc(2000*sizeof(char));
  variables.beam = (char*) malloc(2000*sizeof(char));
  variables.modin = (char*) malloc(2000*sizeof(char));
  variables.path = (char*) malloc(2000*sizeof(char));
  variables.multigpu = 0;
  variables.select = 0;
  variables.blockSizeX = -1;
  variables.blockSizeY = -1;
  variables.blockSizeV = -1;

	long next_op;
	const char* const short_op = "hi:o:I:m:b:M:s:p:X:Y:V:";

	const struct option long_op[] = { //Flag for help
                                    {"help", 0, NULL, 'h' },
                                    /* These options set a flag. */
                                    {"verbose", 0, &verbose_flag, 1},
                                    /* These options don’t set a flag. */
                                    {"input", 1, NULL, 'i' }, {"output", 1, NULL, 'o'},
                                    {"inputdat", 1, NULL, 'I'}, {"modin", 1, NULL, 'm' }, {"beam", 1, NULL, 'b' },
                                    {"multigpu", 1, NULL, 'M'}, {"select", 1, NULL, 's'}, {"path", 1, NULL, 'p'},
                                    {"blockSizeX", 1, NULL, 'X'}, {"blockSizeY", 1, NULL, 'Y'}, {"blockSizeV", 1, NULL, 'V'},
                                    { NULL, 0, NULL, 0 }};

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
		case 'i':
			strcpy(variables.input, optarg);
			break;
    case 'o':
  		strcpy(variables.output, optarg);
  		break;
    case 'I':
      strcpy(variables.inputdat, optarg);
      break;
    case 'm':
    	strcpy(variables.modin, optarg);
    	break;
    case 'b':
    	strcpy(variables.beam, optarg);
    	break;
    case 'p':
      strcpy(variables.path, optarg);
      break;
    case 'M':
      variables.multigpu = atoi(optarg);
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

  if(variables.blockSizeX == -1 || variables.blockSizeY == -1 || variables.blockSizeV == -1 ||
     variables.input == "" || variables.output == "" || variables.inputdat == "" ||
     variables.beam == "" || variables.modin == "" || variables.path == "") {
        print_help();
        exit(EXIT_FAILURE);
  }

  if(!isPow2(variables.blockSizeX) || !isPow2(variables.blockSizeY) || !isPow2(variables.blockSizeV)){
    print_help();
    exit(EXIT_FAILURE);
  }

  if(variables.multigpu != 0 && variables.select != 0){
    print_help();
    exit(EXIT_FAILURE);
  }
	return variables;
}

__host__ void toFitsFloat(cufftComplex *I, int iteration, long M, long N, int option)
{
	fitsfile *fpointer;
	int status = 0;
	long fpixel = 1;
	long elements = M*N;
	char name[60]="";
	long naxes[2]={M,N};
	long naxis = 2;
  char *unit = "JY/PIXEL";
  switch(option){
    case 0:
      sprintf(name, "!%smod_out.fits", mempath, iteration);
      break;
    case 1:
      sprintf(name, "!%sMEM_%d.fits", mempath, iteration);
      break;
    case 2:
      sprintf(name, "!%sMEM_V_%d.fits", mempath, iteration);
      break;
    case 3:
      sprintf(name, "!%sMEM_VB_%d.fits", mempath, iteration);
      break;
    case 4:
      sprintf(name, "!%satten_%d.fits", mempath, iteration);
      break;
    case 5:
      sprintf(name, "!%stotal_atten_0.fits", mempath, iteration);
      break;
    case 6:
      sprintf(name, "!%snoise_0.fits", mempath, iteration);
      break;
    case -1:
      break;
    default:
      printf("Invalid case to FITS\n");
      goToError();
  }


	fits_create_file(&fpointer, name, &status);
  if (status) {
    fits_report_error(stderr, status); /* print error message */
    goToError();
  }
  fits_copy_header(mod_in, fpointer, &status);
  if (status) {
    fits_report_error(stderr, status); /* print error message */
    goToError();
  }
  if(option==0){
    fits_update_key(fpointer, TSTRING, "BUNIT", unit, "Unit of measurement", &status);
  }
  cufftComplex *host_IFITS;
  host_IFITS = (cufftComplex*)malloc(M*N*sizeof(cufftComplex));
  gpuErrchk(cudaMemcpy2D(host_IFITS, sizeof(cufftComplex), I, sizeof(cufftComplex), sizeof(cufftComplex), M*N, cudaMemcpyDeviceToHost));

	float* image2D;
	image2D = (float*) malloc(M*N*sizeof(float));

  int x = M-1;
  int y = N-1;
  for(int i=0; i < M; i++){
		for(int j=0; j < N; j++){
      if(option == 0){
			  image2D[N*y+x] = host_IFITS[N*i+j].x * fg_scale;
      }else if (option == 2 || option == 3){
        image2D[N*y+x] = sqrt(host_IFITS[N*i+j].x * host_IFITS[N*i+j].x + host_IFITS[N*i+j].y * host_IFITS[N*i+j].y);
        //image2D[N*x+y] = host_IFITS[N*i+j].y;
      }else if(option == 4 || option == 5 || option == 6){
        image2D[N*i+j] = host_IFITS[N*i+j].x;
      }else{
        image2D[N*y+x] = host_IFITS[N*i+j].x;
      }
      x--;
		}
    x=M-1;
    y--;
	}

	fits_write_img(fpointer, TFLOAT, fpixel, elements, image2D, &status);
  if (status) {
    fits_report_error(stderr, status); /* print error message */
    goToError();
  }
	fits_close_file(fpointer, &status);
  if (status) {
    fits_report_error(stderr, status); /* print error message */
    goToError();
  }

  free(host_IFITS);
	free(image2D);
}


template <bool nIsPow2>
__global__ void deviceReduceKernel(float *g_idata, float *g_odata, unsigned int n)
{
    extern __shared__ float sdata[];

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int blockSize = blockDim.x;
    unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;

    float mySum = 0.f;

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
    __syncthreads();


    // do reduction in shared mem
    if ((blockSize >= 512) && (tid < 256))
    {
        sdata[tid] = mySum = mySum + sdata[tid + 256];
    }

    __syncthreads();

    if ((blockSize >= 256) &&(tid < 128))
    {
            sdata[tid] = mySum = mySum + sdata[tid + 128];
    }

     __syncthreads();

    if ((blockSize >= 128) && (tid <  64))
    {
       sdata[tid] = mySum = mySum + sdata[tid +  64];
    }

    __syncthreads();

#if (__CUDA_ARCH__ >= 300 )
    if ( tid < 32 )
    {
        // Fetch final intermediate sum from 2nd warp
        if (blockSize >=  64) mySum += sdata[tid + 32];
        // Reduce final warp using shuffle
        for (int offset = warpSize/2; offset > 0; offset /= 2)
        {
            mySum += __shfl_down(mySum, offset);
        }
    }
#else
    // fully unroll reduction within a single warp
    if ((blockSize >=  64) && (tid < 32))
    {
        sdata[tid] = mySum = mySum + sdata[tid + 32];
    }

    __syncthreads();

    if ((blockSize >=  32) && (tid < 16))
    {
        sdata[tid] = mySum = mySum + sdata[tid + 16];
    }

    __syncthreads();

    if ((blockSize >=  16) && (tid <  8))
    {
        sdata[tid] = mySum = mySum + sdata[tid +  8];
    }

    __syncthreads();

    if ((blockSize >=   8) && (tid <  4))
    {
        sdata[tid] = mySum = mySum + sdata[tid +  4];
    }

    __syncthreads();

    if ((blockSize >=   4) && (tid <  2))
    {
        sdata[tid] = mySum = mySum + sdata[tid +  2];
    }

    __syncthreads();

    if ((blockSize >=   2) && ( tid <  1))
    {
        sdata[tid] = mySum = mySum + sdata[tid +  1];
    }

    __syncthreads();
#endif

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = mySum;
}





__host__ float deviceReduce(float *in, long N) {
  float *device_out;
  gpuErrchk(cudaMalloc(&device_out, sizeof(float)*1024));
  gpuErrchk(cudaMemset(device_out, 0, sizeof(float)*1024));

  int threads = 512;
  int blocks = min((int(NearestPowerOf2(N)) + threads - 1) / threads, 1024);
  int smemSize = (threads <= 32) ? 2 * threads * sizeof(float) : threads * sizeof(float);

  bool isPower2 = isPow2(N);
  if(isPower2){
    deviceReduceKernel<true><<<blocks, threads, smemSize>>>(in, device_out, N);
    gpuErrchk(cudaDeviceSynchronize());
  }else{
    deviceReduceKernel<false><<<blocks, threads, smemSize>>>(in, device_out, N);
    gpuErrchk(cudaDeviceSynchronize());
  }

  float *h_odata = (float *) malloc(blocks*sizeof(float));
  float sum = 0.0;

  gpuErrchk(cudaMemcpy(h_odata, device_out, blocks * sizeof(float),cudaMemcpyDeviceToHost));
  for (int i=0; i<blocks; i++)
  {
    sum += h_odata[i];
  }
  cudaFree(device_out);
  free(h_odata);
	return sum;
}

__global__ void hermitianSymmetry(float *Ux, float *Vx, cufftComplex *Vo, float freq, int numVisibilities)
{
  int i = threadIdx.x + blockDim.x * blockIdx.x;

  if (i < numVisibilities){
      if(Ux[i] < 0.0){
        Ux[i] *= -1.0;
        Vx[i] *= -1.0;
        Vo[i].y *= -1.0;
      }
      Ux[i] = (Ux[i] * freq) / LIGHTSPEED;
      Vx[i] = (Vx[i] * freq) / LIGHTSPEED;
  }
}

__global__ void backUV(float *Ux, float *Vx, float freq, int numVisibilities)
{
  int i = threadIdx.x + blockDim.x * blockIdx.x;

  if (i < numVisibilities){
      Ux[i] = (Ux[i] * LIGHTSPEED) / freq;
      Vx[i] = (Vx[i] * LIGHTSPEED) / freq;
  }
}

__global__ void attenuation(cufftComplex *attenMatrix, float frec, long N, float xobs, float yobs, float DELTAX, float DELTAY)
{

		int j = threadIdx.x + blockDim.x * blockIdx.x;
		int i = threadIdx.y + blockDim.y * blockIdx.y;

    int x0 = xobs;
    int y0 = yobs;
    float x = (j - x0) * DELTAX * RPDEG;
    float y = (i - y0) * DELTAY * RPDEG;

    float arc = sqrtf(x*x+y*y);
    float c = 4.0*logf(2.0);
    //printf("frec:%f\n", frec);
    float a = (FWHM*BEAM_FREQ/(frec*1e-9));
    float r = arc/a;
    float atten = expf(-c*r*r);
    attenMatrix[N*i+j].x = atten;
    attenMatrix[N*i+j].y = 0;
}



__global__ void total_attenuation(cufftComplex *total_atten, cufftComplex *attenperFreq, long N)
{
  int j = threadIdx.x + blockDim.x * blockIdx.x;
  int i = threadIdx.y + blockDim.y * blockIdx.y;

  total_atten[N*i+j].x += attenperFreq[N*i+j].x;
  total_atten[N*i+j].y = 0;
}

__global__ void mean_attenuation(cufftComplex *total_atten, int channels, long N)
{
  int j = threadIdx.x + blockDim.x * blockIdx.x;
  int i = threadIdx.y + blockDim.y * blockIdx.y;

  total_atten[N*i+j].x /= channels;
  total_atten[N*i+j].y = 0;
}

__global__ void noise_image(cufftComplex *total_atten, cufftComplex *noise_image, float difmap_noise, long N)
{
  int j = threadIdx.x + blockDim.x * blockIdx.x;
  int i = threadIdx.y + blockDim.y * blockIdx.y;

  float weight = 0.0;
  float noiseval = 0.0;
  float atten = total_atten[N*i+j].x;
  weight = (atten / difmap_noise) * (atten / difmap_noise);
  noiseval = sqrtf(1.0/weight);
  noise_image[N*i+j].x = noiseval;
  noise_image[N*i+j].y = 0;
}

__global__ void apply_beam(cufftComplex *image, cufftComplex *fg_image, long N, float xobs, float yobs, float fg_scale, float frec, float DELTAX, float DELTAY)
{
    int j = threadIdx.x + blockDim.x * blockIdx.x;
    int i = threadIdx.y + blockDim.y * blockIdx.y;


    float dx = DELTAX * 60.0;
    float dy = DELTAY * 60.0;
    float x = (j - xobs) * dx;
    float y = (i - yobs) * dy;
    float arc = RPARCM*sqrtf(x*x+y*y);
    float c = 4.0*logf(2.0);
    float a = (FWHM*BEAM_FREQ/(frec*1e-9));
    float r = arc/a;
    float atten = expf(-c*r*r);

    image[N*i+j].x = fg_image[N*i+j].x * fg_scale * atten;
    image[N*i+j].y = 0.f;
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

    float u,v;
    float du = xphs/M;
    float dv = yphs/N;

    if(j < M/2){
      u = du * j;
    }else{
      u = du * (j-M);
    }

    if(i < N/2){
      v = dv * i;
    }else{
      v = dv * (i-N);
    }

    float phase = -2.0*(u+v);
    float c, s;
    #if (__CUDA_ARCH__ >= 300 )
      sincospif(phase, &s, &c);
    #else
      c = cospif(phase);
      s = sinpif(phase);
    #endif
    float  re = data[N*i+j].x;
    float im = data[N*i+j].y;
    data[N*i+j].x = re * c - im * s;
    data[N*i+j].y = re * s + im * c;
}


/*
 * Interpolate in the visibility array to find the visibility at (u,v);
 */
__global__ void residual(cufftComplex *Vr, cufftComplex *Vo, cufftComplex *V, float *Ux, float *Vx, float deltau, float deltav, long numVisibilities, long N)
{
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  long i1, i2, j1, j2;
  float du, dv;
  float v11, v12, v21, v22;
  float Zreal;
  float Zimag;
  if (i < numVisibilities){
    float u = Ux[i]/deltau;
    float v = Vx[i]/deltav;

    if (fabsf(u) > (N/2)+0.5 || fabsf(v) > (N/2)+0.5) {
      printf("Error in residual: u,v = %f,%f\n", u, v);
      asm("trap;");
    }

    if(u < 0.0){
      u = N + u;
    }

    if(v < 0.0){
      v = N + v;
    }

    i1 = u;
    i2 = (i1+1)%N;
    du = u - i1;
    j1 = v;
    j2 = (j1+1)%N;
    dv = v - j1;

    if (i1 < 0 || i1 > N || j1 < 0 || j2 > N) {
      printf("Error in residual: u,v = %f,%f, %ld,%ld, %ld,%ld\n", u, v, i1, i2, j1, j2);
      asm("trap;");
    }

    /*if(i1 == 511 || i2 == 0){
      printf("Positions (%ld,%ld); (%ld, %ld); (%ld, %ld); (%ld, %ld)\n", i1, j1, i1, j2, i2, j1, i2, j2);
    }*/
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

    Vr[i].x =  Zreal - Vo[i].x;
    Vr[i].y =  Zimag - Vo[i].y;

  }

}



__global__ void clipWNoise(cufftComplex *fg_image, cufftComplex *noise, cufftComplex *I, long N, float noise_cut, float MINPIX)
{
	int j = threadIdx.x + blockDim.x * blockIdx.x;
	int i = threadIdx.y + blockDim.y * blockIdx.y;


  if(noise[N*i+j].x > noise_cut){
    I[N*i+j].x = MINPIX;
  }

  fg_image[N*i+j].x = I[N*i+j].x;
  //printf("%f\n", fg_image[N*i+j].x);
  fg_image[N*i+j].y = 0;
}


__global__ void getGandDGG(float *gg, float *dgg, float *xi, float *g, long N)
{
  int j = threadIdx.x + blockDim.x * blockIdx.x;
	int i = threadIdx.y + blockDim.y * blockIdx.y;

  gg[N*i+j] = g[N*i+j] * g[N*i+j];
  dgg[N*i+j] = (xi[N*i+j] + g[N*i+j]) * xi[N*i+j];
}

__global__ void newP(cufftComplex *p, float *xi, float xmin, float MINPIX, long N)
{
	int j = threadIdx.x + blockDim.x * blockIdx.x;
	int i = threadIdx.y + blockDim.y * blockIdx.y;

  xi[N*i+j] *= xmin;
  if(p[N*i+j].x + xi[N*i+j] > MINPIX){
    p[N*i+j].x += xi[N*i+j];
  }else{
    p[N*i+j].x = MINPIX;
    xi[N*i+j] = 0.0;
  }
  p[N*i+j].y = 0.0;
}

__global__ void newPNoPositivity(cufftComplex *p, float *xi, float xmin, long N)
{
	int j = threadIdx.x + blockDim.x * blockIdx.x;
	int i = threadIdx.y + blockDim.y * blockIdx.y;

  xi[N*i+j] *= xmin;
  p[N*i+j].x += xi[N*i+j];
  p[N*i+j].y = 0.0;
}

__global__ void evaluateXt(cufftComplex *xt, cufftComplex *pcom, float *xicom, float x, float MINPIX, long N)
{
	int j = threadIdx.x + blockDim.x * blockIdx.x;
	int i = threadIdx.y + blockDim.y * blockIdx.y;

  if(pcom[N*i+j].x + x * xicom[N*i+j] > MINPIX){
    xt[N*i+j].x = pcom[N*i+j].x + x * xicom[N*i+j];
  }else{
      xt[N*i+j].x = MINPIX;
  }
  xt[N*i+j].y = 0.0;
}

__global__ void evaluateXtNoPositivity(cufftComplex *xt, cufftComplex *pcom, float *xicom, float x, long N)
{
	int j = threadIdx.x + blockDim.x * blockIdx.x;
	int i = threadIdx.y + blockDim.y * blockIdx.y;

  xt[N*i+j].x = pcom[N*i+j].x + x * xicom[N*i+j];
  xt[N*i+j].y = 0.0;
}


__global__ void chi2Vector(float *chi2, cufftComplex *Vr, float *w, long numVisibilities){
	int i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i < numVisibilities){
		chi2[i] =  w[i] * ((Vr[i].x * Vr[i].x) + (Vr[i].y * Vr[i].y));
	}

}

__global__ void HVector(float *H, cufftComplex *noise, cufftComplex *I, long N, float noise_cut, float MINPIX)
{
	int j = threadIdx.x + blockDim.x * blockIdx.x;
	int i = threadIdx.y + blockDim.y * blockIdx.y;

  float entropy = 0.0;
  if(noise[N*i+j].x <= noise_cut){
    entropy = I[N*i+j].x * logf(I[N*i+j].x / MINPIX);
  }

  H[N*i+j] = entropy;
}

__global__ void QVector(float *H, cufftComplex *noise, cufftComplex *I, long N, float noise_cut, float MINPIX)
{
	int j = threadIdx.x + blockDim.x * blockIdx.x;
	int i = threadIdx.y + blockDim.y * blockIdx.y;

  float entropy = 0.0;
  if(noise[N*i+j].x <= noise_cut){
    if((i>0 && i<N) && (j>0 && j<N)){
      entropy = (I[N*i+j].x - I[N*i+(j-1)].x) * (I[N*i+j].x - I[N*i+(j-1)].x) + (I[N*i+j].x - I[N*i+(j+1)].x) * (I[N*i+j].x - I[N*i+(j+1)].x) + (I[N*i+j].x - I[N*(i-1)+j].x) * (I[N*i+j].x - I[N*(i-1)+j].x) + (I[N*i+j].x - I[N*(i+1)+j].x) * (I[N*i+j].x - I[N*(i+1)+j].x);
      entropy /= 2;
    }else{
      entropy = I[N*i+j].x;
    }
  }

  H[N*i+j] = entropy;
}

__global__ void TVVector(float *TV, cufftComplex *noise, cufftComplex *I, long N, float noise_cut, float MINPIX)
{
	int j = threadIdx.x + blockDim.x * blockIdx.x;
	int i = threadIdx.y + blockDim.y * blockIdx.y;

  float tv = 0.0;
  if(noise[N*i+j].x <= noise_cut){
    if(i!= N-1 || j!=N-1){
      float dx = I[N*i+(j+1)].x - I[N*i+j].x;
      float dy = I[N*(i+1)+j].x - I[N*i+j].x;
      tv = sqrtf((dx * dx) + (dy * dy));
    }else{
      tv = 0;
    }
  }

  TV[N*i+j] = tv;
}
__global__ void searchDirection(float *g, float *xi, float *h, long N)
{
  int j = threadIdx.x + blockDim.x * blockIdx.x;
	int i = threadIdx.y + blockDim.y * blockIdx.y;

  g[N*i+j] = -xi[N*i+j];
  xi[N*i+j] = h[N*i+j] = g[N*i+j];
}

__global__ void newXi(float *g, float *xi, float *h, float gam, long N)
{
  int j = threadIdx.x + blockDim.x * blockIdx.x;
	int i = threadIdx.y + blockDim.y * blockIdx.y;

  g[N*i+j] = -xi[N*i+j];
  xi[N*i+j] = h[N*i+j] = g[N*i+j] + gam * h[N*i+j];
}

__global__ void restartDPhi(float *dphi, float *dChi2, float *dH, long N)
{
  int j = threadIdx.x + blockDim.x * blockIdx.x;
  int i = threadIdx.y + blockDim.y * blockIdx.y;

  dphi[N*i+j] = 0.0;
  dChi2[N*i+j] = 0.0;
  dH[N*i+j] = 0.0;

}

__global__ void DH(float *dH, cufftComplex *I, cufftComplex *noise, float noise_cut, float lambda, float MINPIX, long N)
{
  int j = threadIdx.x + blockDim.x * blockIdx.x;
	int i = threadIdx.y + blockDim.y * blockIdx.y;

  if(noise[N*i+j].x <= noise_cut){
    dH[N*i+j] = lambda * (logf(I[N*i+j].x / MINPIX) + 1.0);
  }
}

__global__ void DQ(float *dQ, cufftComplex *I, cufftComplex *noise, float noise_cut, float lambda, float MINPIX, long N)
{
  int j = threadIdx.x + blockDim.x * blockIdx.x;
	int i = threadIdx.y + blockDim.y * blockIdx.y;

  if(noise[N*i+j].x <= noise_cut){
    if((i>0 && i<N) && (j>0 && j<N)){
    //dQ[N*i+j] = lambda * (logf(I[N*i+j].x / MINPIX) + 1.0);
    dQ[N*i+j] = (I[N*i+j].x - I[N*i+(j-1)].x) + (I[N*i+j].x - I[N*i+(j+1)].x) + (I[N*i+j].x - I[N*(i-1)+j].x)  + (I[N*i+j].x - I[N*(i+1)+j].x);
  }else{
    dQ[N*i+j] = I[N*i+j].x;
    }
    dQ[N*i+j] *= lambda;
  }
}

__global__ void DTV(float *dTV, cufftComplex *I, cufftComplex *noise, float noise_cut, float lambda, float MINPIX, long N)
{
  int j = threadIdx.x + blockDim.x * blockIdx.x;
	int i = threadIdx.y + blockDim.y * blockIdx.y;

  float dtv = 0.0;
  float num = 0.0;
  float den = 0.0;
  if(noise[N*i+j].x <= noise_cut){
    if(i!= N-1 || j!=N-1){
      float a = I[N*i+(j+1)].x;
      float b = I[N*(i+1)+j].x;
      float y = I[N*i+j].x;
      float num = -a-b+(2*y);
      float den = (a*a) - 2*y*(a+b) + (b*b) + 2*(y*y);
      if(den <= 0){
        dtv = MINPIX;
      }else{
        dtv = num/sqrtf(den);
      }
    }else{
      dtv = MINPIX;
    }
    dTV[N*i+j] = lambda * dtv;
  }
}

__global__ void DChi2(cufftComplex *noise, cufftComplex *atten, float *dChi2, cufftComplex *Vr, float *U, float *V, float *w, long N, long numVisibilities, float fg_scale, float noise_cut, float xobs, float yobs, float DELTAX, float DELTAY)
{

	int j = threadIdx.x + blockDim.x * blockIdx.x;
	int i = threadIdx.y + blockDim.y * blockIdx.y;

  int x0 = xobs;
  int y0 = yobs;
  float x = (j - x0) * DELTAX * RPDEG;
  float y = (i - y0) * DELTAY * RPDEG;

	float Ukv;
	float Vkv;

	float cosk;
	float sink;

  float dchi2 = 0.0;
  if(noise[N*i+j].x <= noise_cut){
  	for(int v=0; v<numVisibilities; v++){
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

  dchi2 *= fg_scale * atten[N*i+j].x;
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

__global__ void substraction(float *x, cufftComplex *xc, float *gc, float lambda, long N)
{
  int j = threadIdx.x + blockDim.x * blockIdx.x;
	int i = threadIdx.y + blockDim.y * blockIdx.y;

  x[N*i+j] = xc[N*i+j].x - lambda*gc[N*i+j];
}

__global__ void projection(float *px, float *x, float MINPIX, long N){

  int j = threadIdx.x + blockDim.x * blockIdx.x;
	int i = threadIdx.y + blockDim.y * blockIdx.y;


  if(INFINITY < x[N*i+j]){
    px[N*i+j] = INFINITY;
  }else{
    px[N*i+j] = x[N*i+j];
  }

  if(MINPIX > px[N*i+j]){
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


__host__ float chiCuadrado(cufftComplex *I)
{
  if(num_gpus == 1){
    cudaSetDevice(selected);
  }else{
    cudaSetDevice(0);
  }
  //printf("**************Calculating phi - Iteration %d **************\n", iter);
  float resultPhi = 0.0;
  float resultchi2  = 0.0;
  float resultH  = 0.0;


  clipWNoise<<<numBlocksNN, threadsPerBlockNN>>>(device_fg_image, device_noise_image, I, N, noise_cut, MINPIX);
  gpuErrchk(cudaDeviceSynchronize());


  if(iter>=1 && MINPIX!=0.0){
    HVector<<<numBlocksNN, threadsPerBlockNN>>>(device_H, device_noise_image, device_fg_image, N, noise_cut, MINPIX);
    gpuErrchk(cudaDeviceSynchronize());
  }

  if(num_gpus == 1){
    cudaSetDevice(selected);
    for(int i=0; i<data.total_frequencies;i++){

    	apply_beam<<<numBlocksNN, threadsPerBlockNN>>>(device_image, device_fg_image, N, global_xobs, global_yobs, fg_scale, visibilities[i].freq, DELTAX, DELTAY);
    	gpuErrchk(cudaDeviceSynchronize());

    	//FFT 2D
    	if ((cufftExecC2C(plan1GPU, (cufftComplex*)device_image, (cufftComplex*)device_V, CUFFT_FORWARD)) != CUFFT_SUCCESS) {
    		printf("CUFFT exec error\n");
    		goToError();
    	}
    	gpuErrchk(cudaDeviceSynchronize());

      //PHASE_ROTATE VISIBILITIES
      phase_rotate<<<numBlocksNN, threadsPerBlockNN>>>(device_V, M, N, global_xobs, global_yobs);
    	gpuErrchk(cudaDeviceSynchronize());

      //RESIDUAL CALCULATION
      residual<<<visibilities[i].numBlocksUV, visibilities[i].threadsPerBlockUV>>>(device_visibilities[i].Vr, device_visibilities[i].Vo, device_V, device_visibilities[i].u, device_visibilities[i].v, deltau, deltav, data.numVisibilitiesPerFreq[i], N);
    	gpuErrchk(cudaDeviceSynchronize());


    	////chi 2 VECTOR
    	chi2Vector<<<visibilities[i].numBlocksUV, visibilities[i].threadsPerBlockUV>>>(device_vars[i].chi2, device_visibilities[i].Vr, device_visibilities[i].weight, data.numVisibilitiesPerFreq[i]);
    	gpuErrchk(cudaDeviceSynchronize());

    	//REDUCTIONS
    	//chi2
    	resultchi2  += deviceReduce(device_vars[i].chi2, data.numVisibilitiesPerFreq[i]);
      //printf("resultchi2: %f\n", resultchi2);
    }
  }else{
    #pragma omp parallel for schedule(static,1)
    for (int i = 0; i < data.total_frequencies; i++)
		{
      float result = 0.0;
      unsigned int j = omp_get_thread_num();
			//unsigned int num_cpu_threads = omp_get_num_threads();
			// set and check the CUDA device for this CPU thread
			int gpu_id = -1;
			cudaSetDevice(i % num_gpus);   // "% num_gpus" allows more CPU threads than GPU devices
			cudaGetDevice(&gpu_id);

    	apply_beam<<<numBlocksNN, threadsPerBlockNN>>>(device_vars[i].device_image, device_fg_image, N, global_xobs, global_yobs, fg_scale, visibilities[i].freq, DELTAX, DELTAY);
    	gpuErrchk(cudaDeviceSynchronize());

    	//FFT 2D
    	if ((cufftExecC2C(device_vars[i].plan, (cufftComplex*)device_vars[i].device_image, (cufftComplex*)device_vars[i].device_V, CUFFT_FORWARD)) != CUFFT_SUCCESS) {
    		printf("CUFFT exec error\n");
    		//return -1 ;
    		goToError();
    	}
    	gpuErrchk(cudaDeviceSynchronize());

      //PHASE_ROTATE VISIBILITIES
      phase_rotate<<<numBlocksNN, threadsPerBlockNN>>>(device_vars[i].device_V, M, N, global_xobs, global_yobs);
    	gpuErrchk(cudaDeviceSynchronize());

      //RESIDUAL CALCULATION
      residual<<<visibilities[i].numBlocksUV, visibilities[i].threadsPerBlockUV>>>(device_visibilities[i].Vr, device_visibilities[i].Vo, device_vars[i].device_V, device_visibilities[i].u, device_visibilities[i].v, deltau, deltav, data.numVisibilitiesPerFreq[i], N);
    	gpuErrchk(cudaDeviceSynchronize());


    	////chi2 VECTOR
    	chi2Vector<<<visibilities[i].numBlocksUV, visibilities[i].threadsPerBlockUV>>>(device_vars[i].chi2, device_visibilities[i].Vr, device_visibilities[i].weight, data.numVisibilitiesPerFreq[i]);
    	gpuErrchk(cudaDeviceSynchronize());


      result = deviceReduce(device_vars[i].chi2, data.numVisibilitiesPerFreq[i]);
    	//REDUCTIONS
    	//chi2
      #pragma omp critical
      {
        resultchi2  += result;
      }

    }
  }
    if(num_gpus == 1){
      cudaSetDevice(selected);
    }else{
      cudaSetDevice(0);
    }
    resultH  = deviceReduce(device_H, M*N);
    resultPhi = (0.5 * resultchi2) + (lambda * resultH);
    /*printf("chi2 value = %.5f\n", resultchi2);
    printf("H value = %.5f\n", resultH);
    printf("(1/2) * chi2 value = %.5f\n", 0.5*resultchi2);
    printf("lambda * H value = %.5f\n", lambda*resultH);
    printf("Phi value = %.5f\n\n", resultPhi);*/

  	return resultPhi;
}



__host__ void dchiCuadrado(cufftComplex *I, float *dxi2)
{
	//printf("**************Calculating dphi - Iteration %d *************\n", iter);


  if(num_gpus == 1){
    cudaSetDevice(selected);
  }else{
    cudaSetDevice(0);
  }

  restartDPhi<<<numBlocksNN, threadsPerBlockNN>>>(device_dphi, device_dchi2_total, device_H, N);
  gpuErrchk(cudaDeviceSynchronize());


  toFitsFloat(I, iter, M, N, 1);
  //toFitsFloat(device_V, iter, M, N, 2);

  if(iter >= 1 && MINPIX!=0.0){

    DH<<<numBlocksNN, threadsPerBlockNN>>>(device_dH, I, device_noise_image, noise_cut, lambda, MINPIX, N);
    gpuErrchk(cudaDeviceSynchronize());

  }

  if(num_gpus == 1){
    cudaSetDevice(selected);
    for(int i=0; i<data.total_frequencies;i++){

        DChi2<<<numBlocksNN, threadsPerBlockNN>>>(device_noise_image, device_vars[i].atten, device_vars[i].dchi2, device_visibilities[i].Vr, device_visibilities[i].u, device_visibilities[i].v, device_visibilities[i].weight, N, data.numVisibilitiesPerFreq[i], fg_scale, noise_cut, global_xobs, global_yobs, DELTAX, DELTAY);
      	gpuErrchk(cudaDeviceSynchronize());

        DChi2_total<<<numBlocksNN, threadsPerBlockNN>>>(device_dchi2_total, device_vars[i].dchi2, N);
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
      DChi2<<<numBlocksNN, threadsPerBlockNN>>>(device_noise_image, device_vars[i].atten, device_vars[i].dchi2, device_visibilities[i].Vr, device_visibilities[i].u, device_visibilities[i].v, device_visibilities[i].weight, N, data.numVisibilitiesPerFreq[i], fg_scale, noise_cut, global_xobs, global_yobs, DELTAX, DELTAY);
      gpuErrchk(cudaDeviceSynchronize());

      #pragma omp critical
      {
        DChi2_total<<<numBlocksNN, threadsPerBlockNN>>>(device_dchi2_total, device_vars[i].dchi2, N);
        gpuErrchk(cudaDeviceSynchronize());
      }

    }
  }

  if(num_gpus == 1){
    cudaSetDevice(selected);
  }else{
    cudaSetDevice(0);
  }

  DPhi<<<numBlocksNN, threadsPerBlockNN>>>(device_dphi, device_dchi2_total, device_dH, N);
  gpuErrchk(cudaDeviceSynchronize());



  //dxi2 = device_dphi;
  gpuErrchk(cudaMemcpy2D(dxi2, sizeof(float), device_dphi, sizeof(float), sizeof(float), M*N, cudaMemcpyDeviceToDevice));

}
