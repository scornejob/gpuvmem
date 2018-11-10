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

#include "MSFITSIO.cuh"

__host__ freqData countVisibilities(char * MS_name, Field *&fields)
{
        freqData freqsAndVisibilities;
        string dir = MS_name;
        char *query;
        casa::Vector<double> pointing;
        casa::Table main_tab(dir);
        casa::Table field_tab(main_tab.keywordSet().asTable("FIELD"));
        casa::Table spectral_window_tab(main_tab.keywordSet().asTable("SPECTRAL_WINDOW"));
        casa::Table polarization_tab(main_tab.keywordSet().asTable("POLARIZATION"));
        freqsAndVisibilities.nfields = field_tab.nrow();
        casa::ROTableRow field_row(field_tab, casa::stringToVector("REFERENCE_DIR,NAME"));

        fields = (Field*)malloc(freqsAndVisibilities.nfields*sizeof(Field));
        for(int f=0; f<freqsAndVisibilities.nfields; f++) {
                const casa::TableRecord &values = field_row.get(f);
                pointing = values.asArrayDouble("REFERENCE_DIR");
                fields[f].obsra = pointing[0];
                fields[f].obsdec = pointing[1];
        }

        freqsAndVisibilities.nsamples = main_tab.nrow();
        if (freqsAndVisibilities.nsamples == 0) {
                cout << "ERROR : nsamples is zero... exiting...." << endl;
                exit(-1);
        }

        casa::ROArrayColumn<casa::Double> chan_freq_col(spectral_window_tab,"CHAN_FREQ"); //NUMBER OF SPW
        freqsAndVisibilities.n_internal_frequencies = spectral_window_tab.nrow();

        freqsAndVisibilities.channels = (int*)malloc(freqsAndVisibilities.n_internal_frequencies*sizeof(int));
        casa::ROScalarColumn<casa::Int> n_chan_freq(spectral_window_tab,"NUM_CHAN");
        for(int i = 0; i < freqsAndVisibilities.n_internal_frequencies; i++) {
                freqsAndVisibilities.channels[i] = n_chan_freq(i);
        }

        // We consider all chans .. The data will be processed this way later.

        int total_frequencies = 0;
        for(int i=0; i <freqsAndVisibilities.n_internal_frequencies; i++) {
                for(int j=0; j < freqsAndVisibilities.channels[i]; j++) {
                        total_frequencies++;
                }
        }

        freqsAndVisibilities.total_frequencies = total_frequencies;
        for(int f=0; f < freqsAndVisibilities.nfields; f++) {
                fields[f].numVisibilitiesPerFreq = (long*)malloc(freqsAndVisibilities.total_frequencies*sizeof(long));
                for(int i = 0; i < freqsAndVisibilities.total_frequencies; i++) {
                        fields[f].numVisibilitiesPerFreq[i] = 0;
                }
        }

        casa::ROScalarColumn<casa::Int> n_corr(polarization_tab,"NUM_CORR");
        freqsAndVisibilities.nstokes=n_corr(0);

        casa::Vector<float> weights;
        casa::Matrix<casa::Bool> flagCol;

        bool flag;
        int counter;
        size_t needed;

        // Iteration through all fields

        for(int f=0; f<freqsAndVisibilities.nfields; f++) {
                counter = 0;
                for(int i=0; i < freqsAndVisibilities.n_internal_frequencies; i++) {
                        // Query for data with forced IF and FIELD
                        needed = snprintf(NULL, 0, "select * from %s where DATA_DESC_ID=%d and FIELD_ID=%d and FLAG_ROW=FALSE", MS_name, i,f) + 1;
                        query = (char*) malloc(needed*sizeof(char));
                        snprintf(query, needed, "select * from %s where DATA_DESC_ID=%d and FIELD_ID=%d and FLAG_ROW=FALSE", MS_name, i,f);

                        casa::Table query_tab = casa::tableCommand(query);

                        casa::ROArrayColumn<float> weight_col(query_tab,"WEIGHT");
                        casa::ROArrayColumn<bool> flag_data_col(query_tab,"FLAG");

                        for (int k=0; k < query_tab.nrow(); k++) {
                                flagCol=flag_data_col(k);
                                weights=weight_col(k);
                                for(int j=0; j < freqsAndVisibilities.channels[i]; j++) {
                                        for (int sto=0; sto<freqsAndVisibilities.nstokes; sto++) {
                                                if(flagCol(sto,j) == false && weights[sto] > 0.0) {
                                                        fields[f].numVisibilitiesPerFreq[counter+j]++;
                                                }
                                        }
                                }
                        }
                        counter+=freqsAndVisibilities.channels[i];
                        free(query);
                }
        }

        int local_max = 0;
        int max = 0;
        for(int f=0; f < freqsAndVisibilities.nfields; f++) {
                local_max = *std::max_element(fields[f].numVisibilitiesPerFreq,fields[f].numVisibilitiesPerFreq+total_frequencies);
                if(local_max > max) {
                        max = local_max;
                }
        }
        freqsAndVisibilities.max_number_visibilities_in_channel = max;

        return freqsAndVisibilities;
}


__host__ canvasVariables readCanvas(char *canvas_name, fitsfile *&canvas, float b_noise_aux, int status_canvas, int verbose_flag)
{
        status_canvas = 0;
        int status_noise = 0;

        canvasVariables c_vars;

        fits_open_file(&canvas, canvas_name, 0, &status_canvas);
        if (status_canvas) {
                fits_report_error(stderr, status_canvas); /* print error message */
                exit(0);
        }

        fits_read_key(canvas, TFLOAT, "CDELT1", &c_vars.DELTAX, NULL, &status_canvas);
        fits_read_key(canvas, TFLOAT, "CDELT2", &c_vars.DELTAY, NULL, &status_canvas);
        fits_read_key(canvas, TDOUBLE, "CRVAL1", &c_vars.ra, NULL, &status_canvas);
        fits_read_key(canvas, TDOUBLE, "CRVAL2", &c_vars.dec, NULL, &status_canvas);
        fits_read_key(canvas, TINT, "CRPIX1", &c_vars.crpix1, NULL, &status_canvas);
        fits_read_key(canvas, TINT, "CRPIX2", &c_vars.crpix2, NULL, &status_canvas);
        fits_read_key(canvas, TLONG, "NAXIS1", &c_vars.M, NULL, &status_canvas);
        fits_read_key(canvas, TLONG, "NAXIS2", &c_vars.N, NULL, &status_canvas);
        fits_read_key(canvas, TFLOAT, "BMAJ", &c_vars.beam_bmaj, NULL, &status_canvas);
        fits_read_key(canvas, TFLOAT, "BMIN", &c_vars.beam_bmin, NULL, &status_canvas);
        fits_read_key(canvas, TFLOAT, "NOISE", &c_vars.beam_noise, NULL, &status_noise);

        if (status_canvas) {
                fits_report_error(stderr, status_canvas); /* print error message */
                exit(0);
        }

        if(status_noise) {
                c_vars.beam_noise = b_noise_aux;
        }

        c_vars.beam_bmaj = c_vars.beam_bmaj/ -c_vars.DELTAX;
        c_vars.beam_bmin = c_vars.beam_bmin/ -c_vars.DELTAX;

        if(verbose_flag) {
                cout << "FITS Files READ" << endl;
        }

        return c_vars;
}

__host__ void readFITSImageValues(char *imageName, fitsfile *file, float *&values, int status, long M, long N)
{

        int anynull;
        long fpixel = 1;
        float null = 0.;
        long elementsImage = M*N;

        values = (float*)malloc(M*N*sizeof(float));
        fits_open_file(&file, imageName, 0, &status);
        fits_read_img(file, TFLOAT, fpixel, elementsImage, &null, values, &anynull, &status);

}

__host__ void readMSMCNoise(char *MS_name, Field *fields, freqData data)
{

        char *error = 0;
        int g = 0, h = 0;
        long c;
        char *query;
        string dir = MS_name;
        casa::Table main_tab(dir);
        casa::Table spectral_window_tab(main_tab.keywordSet().asTable("SPECTRAL_WINDOW"));
        casa::Table polarization_tab(main_tab.keywordSet().asTable("POLARIZATION"));

        casa::ROArrayColumn<casa::Int> correlation_col(polarization_tab,"CORR_TYPE");
        casa::Vector<int> polarizations;
        polarizations=correlation_col(0);

        casa::ROArrayColumn<casa::Double> chan_freq_col(spectral_window_tab,"CHAN_FREQ");

        casa::Vector<float> weights;
        casa::Vector<double> uvw;
        casa::Matrix<casa::Complex> dataCol;
        casa::Matrix<casa::Bool> flagCol;

        bool flag;
        size_t needed;

        for(int f=0; f < data.nfields; f++) {
                for(int i = 0; i < data.total_frequencies; i++) {
                        fields[f].numVisibilitiesPerFreq[i] = 0;
                }
        }

        float u;
        SelectStream(0);
        PutSeed(-1);

        for(int f=0; f<data.nfields; f++) {
                g=0;
                for(int i=0; i < data.n_internal_frequencies; i++) {
                        needed = snprintf(NULL, 0, "select * from %s where DATA_DESC_ID=%d and FIELD_ID=%d and FLAG_ROW=FALSE", MS_name, i,f) + 1;
                        query = (char*) malloc(needed*sizeof(char));
                        snprintf(query, needed, "select * from %s where DATA_DESC_ID=%d and FIELD_ID=%d and FLAG_ROW=FALSE", MS_name, i,f);

                        casa::Table query_tab = casa::tableCommand(query);

                        casa::ROArrayColumn<double> uvw_col(query_tab,"UVW");
                        casa::ROArrayColumn<float> weight_col(query_tab,"WEIGHT");
                        casa::ROArrayColumn<casa::Complex> data_col(query_tab,"DATA");
                        casa::ROArrayColumn<bool> flag_data_col(query_tab,"FLAG");
                        for (int k=0; k < query_tab.nrow(); k++) {
                                uvw = uvw_col(k);
                                dataCol = data_col(k);
                                flagCol = flag_data_col(k);
                                weights = weight_col(k);
                                for(int j=0; j < data.channels[i]; j++) {
                                        for (int sto=0; sto<data.nstokes; sto++) {
                                                if(flagCol(sto,j) == false && weights[sto] > 0.0) {
                                                        c = fields[f].numVisibilitiesPerFreq[g+j];
                                                        fields[f].visibilities[g+j].stokes[c] = polarizations[sto];
                                                        fields[f].visibilities[g+j].u[c] = uvw[0];
                                                        fields[f].visibilities[g+j].v[c] = uvw[1];
                                                        u = Normal(0.0, 1.0);
                                                        fields[f].visibilities[g+j].Vo[c].x = dataCol(sto,j).real() + u * (1/sqrt(weights[sto]));
                                                        u = Normal(0.0, 1.0);
                                                        fields[f].visibilities[g+j].Vo[c].y = dataCol(sto,j).imag() + u * (1/sqrt(weights[sto]));
                                                        fields[f].visibilities[g+j].weight[c] = weights[sto];
                                                        fields[f].numVisibilitiesPerFreq[g+j]++;
                                                }
                                        }
                                }
                        }
                        g+=data.channels[i];
                        free(query);
                }
        }


        for(int f=0; f<data.nfields; f++) {
                h = 0;
                for(int i = 0; i < data.n_internal_frequencies; i++) {
                        casa::Vector<double> chan_freq_vector;
                        chan_freq_vector=chan_freq_col(i);
                        for(int j = 0; j < data.channels[i]; j++) {
                                fields[f].visibilities[h].freq = chan_freq_vector[j];
                                h++;
                        }
                }
        }

        for(int f=0; f<data.nfields; f++) {
                h = 0;
                fields[f].valid_frequencies = 0;
                for(int i = 0; i < data.n_internal_frequencies; i++) {
                        for(int j = 0; j < data.channels[i]; j++) {
                                if(fields[f].numVisibilitiesPerFreq[h] > 0) {
                                        fields[f].valid_frequencies++;
                                }
                                h++;
                        }
                }
        }
}

__host__ void readSubsampledMS(char *MS_name, Field *fields, freqData data, float random_probability)
{
        char *error = 0;
        int g = 0, h = 0;
        long c;
        char *query;
        string dir = MS_name;
        casa::Table main_tab(dir);
        casa::Table spectral_window_tab(main_tab.keywordSet().asTable("SPECTRAL_WINDOW"));
        casa::Table polarization_tab(main_tab.keywordSet().asTable("POLARIZATION"));

        casa::ROArrayColumn<casa::Int> correlation_col(polarization_tab,"CORR_TYPE");
        casa::Vector<int> polarizations;
        polarizations=correlation_col(0);

        casa::ROArrayColumn<casa::Double> chan_freq_col(spectral_window_tab,"CHAN_FREQ");

        casa::Vector<float> weights;
        casa::Vector<double> uvw;
        casa::Matrix<casa::Complex> dataCol;
        casa::Matrix<casa::Bool> flagCol;

        bool flag;
        size_t needed;

        for(int f=0; f < data.nfields; f++) {
                for(int i = 0; i < data.total_frequencies; i++) {
                        fields[f].numVisibilitiesPerFreq[i] = 0;
                }
        }

        float u;
        SelectStream(0);
        PutSeed(-1);
        for(int f=0; f<data.nfields; f++) {
                g=0;
                for(int i=0; i < data.n_internal_frequencies; i++) {
                        needed = snprintf(NULL, 0, "select * from %s where DATA_DESC_ID=%d and FIELD_ID=%d and FLAG_ROW=FALSE", MS_name, i,f) + 1;
                        query = (char*) malloc(needed*sizeof(char));
                        snprintf(query, needed, "select * from %s where DATA_DESC_ID=%d and FIELD_ID=%d and FLAG_ROW=FALSE", MS_name, i,f);

                        casa::Table query_tab = casa::tableCommand(query);

                        casa::ROArrayColumn<double> uvw_col(query_tab,"UVW");
                        casa::ROArrayColumn<float> weight_col(query_tab,"WEIGHT");
                        casa::ROArrayColumn<casa::Complex> data_col(query_tab,"DATA");
                        casa::ROArrayColumn<bool> flag_data_col(query_tab,"FLAG");

                        for (int k=0; k < query_tab.nrow(); k++) {
                                uvw = uvw_col(k);
                                dataCol = data_col(k);
                                flagCol = flag_data_col(k);
                                weights = weight_col(k);
                                for(int j=0; j < data.channels[i]; j++) {
                                        for (int sto=0; sto < data.nstokes; sto++) {
                                                if(flagCol(sto,j) == false && weights[sto] > 0.0) {
                                                        u = Random();
                                                        if(u<random_probability) {
                                                                c = fields[f].numVisibilitiesPerFreq[g+j];
                                                                fields[f].visibilities[g+j].stokes[c] = polarizations[sto];
                                                                fields[f].visibilities[g+j].u[c] = uvw[0];
                                                                fields[f].visibilities[g+j].v[c] = uvw[1];
                                                                fields[f].visibilities[g+j].Vo[c].x = dataCol(sto,j).real();
                                                                fields[f].visibilities[g+j].Vo[c].y = dataCol(sto,j).imag();
                                                                fields[f].visibilities[g+j].weight[c] = weights[sto];
                                                                fields[f].numVisibilitiesPerFreq[g+j]++;
                                                        }else{
                                                                c = fields[f].numVisibilitiesPerFreq[g+j];
                                                                fields[f].visibilities[g+j].stokes[c] = polarizations[sto];
                                                                fields[f].visibilities[g+j].u[c] = uvw[0];
                                                                fields[f].visibilities[g+j].v[c] = uvw[1];
                                                                fields[f].visibilities[g+j].Vo[c].x = dataCol(sto,j).real();
                                                                fields[f].visibilities[g+j].Vo[c].y = dataCol(sto,j).imag();
                                                                fields[f].visibilities[g+j].weight[c] = 0.0;
                                                                fields[f].numVisibilitiesPerFreq[g+j]++;
                                                        }
                                                }
                                        }
                                }
                        }
                        g+=data.channels[i];
                        free(query);
                }
        }

        for(int f=0; f<data.nfields; f++) {
                h = 0;
                for(int i = 0; i < data.n_internal_frequencies; i++) {
                        casa::Vector<double> chan_freq_vector;
                        chan_freq_vector=chan_freq_col(i);
                        for(int j = 0; j < data.channels[i]; j++) {
                                fields[f].visibilities[h].freq = chan_freq_vector[j];
                                h++;
                        }
                }
        }

        for(int f=0; f<data.nfields; f++) {
                h = 0;
                fields[f].valid_frequencies = 0;
                for(int i = 0; i < data.n_internal_frequencies; i++) {
                        for(int j = 0; j < data.channels[i]; j++) {
                                if(fields[f].numVisibilitiesPerFreq[h] > 0) {
                                        fields[f].valid_frequencies++;
                                }
                                h++;
                        }
                }
        }

}

__host__ void readMCNoiseSubsampledMS(char *MS_name, Field *fields, freqData data, float random_probability)
{
        char *error = 0;
        int g = 0, h = 0;
        long c;
        char *query;
        string dir = MS_name;
        casa::Table main_tab(dir);

        casa::Table spectral_window_tab(main_tab.keywordSet().asTable("SPECTRAL_WINDOW"));
        casa::Table polarization_tab(main_tab.keywordSet().asTable("POLARIZATION"));

        casa::ROArrayColumn<casa::Int> correlation_col(polarization_tab,"CORR_TYPE");
        casa::Vector<int> polarizations;
        polarizations=correlation_col(0);

        casa::ROArrayColumn<casa::Double> chan_freq_col(spectral_window_tab,"CHAN_FREQ");

        casa::Vector<float> weights;
        casa::Vector<double> uvw;
        casa::Matrix<casa::Complex> dataCol;
        casa::Matrix<casa::Bool> flagCol;

        bool flag;
        size_t needed;

        for(int f=0; f < data.nfields; f++) {
                for(int i = 0; i < data.total_frequencies; i++) {
                        fields[f].numVisibilitiesPerFreq[i] = 0;
                }
        }

        float u;
        float nu;
        SelectStream(0);
        PutSeed(-1);
        for(int f=0; f<data.nfields; f++) {
                g=0;
                for(int i=0; i < data.n_internal_frequencies; i++) {
                        needed = snprintf(NULL, 0, "select * from %s where DATA_DESC_ID=%d and FIELD_ID=%d and FLAG_ROW=FALSE", MS_name, i,f) + 1;
                        query = (char*) malloc(needed*sizeof(char));
                        snprintf(query, needed, "select * from %s where DATA_DESC_ID=%d and FIELD_ID=%d and FLAG_ROW=FALSE", MS_name, i,f);

                        casa::Table query_tab = casa::tableCommand(query);

                        casa::ROArrayColumn<double> uvw_col(query_tab,"UVW");
                        casa::ROArrayColumn<float> weight_col(query_tab,"WEIGHT");
                        casa::ROArrayColumn<casa::Complex> data_col(query_tab,"DATA");
                        casa::ROArrayColumn<bool> flag_data_col(query_tab,"FLAG");

                        for (int k=0; k < query_tab.nrow(); k++) {
                                uvw = uvw_col(k);
                                dataCol = data_col(k);
                                flagCol = flag_data_col(k);
                                weights = weight_col(k);
                                for(int j=0; j < data.channels[i]; j++) {
                                        for (int sto=0; sto < data.nstokes; sto++) {
                                                if(flagCol(sto,j) == false && weights[sto] > 0.0) {
                                                        u = Random();
                                                        if(u<random_probability) {
                                                                c = fields[f].numVisibilitiesPerFreq[g+j];
                                                                fields[f].visibilities[g+j].stokes[c] = polarizations[sto];
                                                                fields[f].visibilities[g+j].u[c] = uvw[0];
                                                                fields[f].visibilities[g+j].v[c] = uvw[1];
                                                                nu = Normal(0.0, 1.0);
                                                                fields[f].visibilities[g+j].Vo[c].x = dataCol(sto,j).real() + u * (1/sqrt(weights[sto]));
                                                                nu = Normal(0.0, 1.0);
                                                                fields[f].visibilities[g+j].Vo[c].y = dataCol(sto,j).imag() + u * (1/sqrt(weights[sto]));
                                                                fields[f].visibilities[g+j].weight[c] = weights[sto];
                                                                fields[f].numVisibilitiesPerFreq[g+j]++;
                                                        }else{
                                                                c = fields[f].numVisibilitiesPerFreq[g+j];
                                                                fields[f].visibilities[g+j].stokes[c] = polarizations[sto];
                                                                fields[f].visibilities[g+j].u[c] = uvw[0];
                                                                fields[f].visibilities[g+j].v[c] = uvw[1];
                                                                fields[f].visibilities[g+j].Vo[c].x = dataCol(sto,j).real();
                                                                fields[f].visibilities[g+j].Vo[c].y = dataCol(sto,j).imag();
                                                                fields[f].visibilities[g+j].weight[c] = 0.0;
                                                                fields[f].numVisibilitiesPerFreq[g+j]++;
                                                        }
                                                }
                                        }
                                }
                        }
                        g+=data.channels[i];
                        free(query);
                }
        }

        for(int f=0; f<data.nfields; f++) {
                h = 0;
                for(int i = 0; i < data.n_internal_frequencies; i++) {
                        casa::Vector<double> chan_freq_vector;
                        chan_freq_vector=chan_freq_col(i);
                        for(int j = 0; j < data.channels[i]; j++) {
                                fields[f].visibilities[h].freq = chan_freq_vector[j];
                                h++;
                        }
                }
        }

        for(int f=0; f<data.nfields; f++) {
                h = 0;
                fields[f].valid_frequencies = 0;
                for(int i = 0; i < data.n_internal_frequencies; i++) {
                        for(int j = 0; j < data.channels[i]; j++) {
                                if(fields[f].numVisibilitiesPerFreq[h] > 0) {
                                        fields[f].valid_frequencies++;
                                }
                                h++;
                        }
                }
        }

}


__host__ void readMS(char *MS_name, Field *fields, freqData data)
{

        char *error = 0;
        int g = 0, h = 0;
        long c;
        char *query;
        string dir = MS_name;
        casa::Table main_tab(dir);

        casa::Table spectral_window_tab(main_tab.keywordSet().asTable("SPECTRAL_WINDOW"));

        casa::Table polarization_tab(main_tab.keywordSet().asTable("POLARIZATION"));
        casa::ROArrayColumn<casa::Int> correlation_col(polarization_tab,"CORR_TYPE");
        casa::Vector<int> polarizations;
        polarizations=correlation_col(0);

        casa::ROArrayColumn<casa::Double> chan_freq_col(spectral_window_tab,"CHAN_FREQ");

        casa::Vector<float> weights;
        casa::Vector<double> uvw;
        casa::Matrix<casa::Complex> dataCol;
        casa::Matrix<casa::Bool> flagCol;
        bool flag;
        size_t needed;

        for(int f=0; f < data.nfields; f++) {
                for(int i = 0; i < data.total_frequencies; i++) {
                        fields[f].numVisibilitiesPerFreq[i] = 0;
                }
        }

        for(int f=0; f<data.nfields; f++) {
                g=0;
                for(int i=0; i < data.n_internal_frequencies; i++) {
                        needed = snprintf(NULL, 0, "select * from %s where DATA_DESC_ID=%d and FIELD_ID=%d and FLAG_ROW=FALSE", MS_name, i,f) + 1;
                        query = (char*) malloc(needed*sizeof(char));
                        snprintf(query, needed, "select * from %s where DATA_DESC_ID=%d and FIELD_ID=%d and FLAG_ROW=FALSE", MS_name, i,f);

                        casa::Table query_tab = casa::tableCommand(query);

                        casa::ROArrayColumn<double> uvw_col(query_tab,"UVW");
                        casa::ROArrayColumn<float> weight_col(query_tab,"WEIGHT");
                        casa::ROArrayColumn<casa::Complex> data_col(query_tab,"DATA");
                        casa::ROArrayColumn<bool> flag_data_col(query_tab,"FLAG");

                        for (int k=0; k < query_tab.nrow(); k++) {
                                uvw = uvw_col(k);
                                dataCol = data_col(k);
                                flagCol = flag_data_col(k);
                                weights = weight_col(k);
                                for(int j=0; j < data.channels[i]; j++) {
                                        for (int sto=0; sto < data.nstokes; sto++) {
                                                if(flagCol(sto,j) == false && weights[sto] > 0.0) {
                                                        c = fields[f].numVisibilitiesPerFreq[g+j];
                                                        fields[f].visibilities[g+j].stokes[c] = polarizations[sto];
                                                        fields[f].visibilities[g+j].u[c] = uvw[0];
                                                        fields[f].visibilities[g+j].v[c] = uvw[1];
                                                        fields[f].visibilities[g+j].Vo[c].x = dataCol(sto,j).real();
                                                        fields[f].visibilities[g+j].Vo[c].y = dataCol(sto,j).imag();
                                                        fields[f].visibilities[g+j].weight[c] = weights[sto];
                                                        fields[f].numVisibilitiesPerFreq[g+j]++;
                                                }
                                        }
                                }
                        }
                        g+=data.channels[i];
                        free(query);
                }
        }


        for(int f=0; f<data.nfields; f++) {
                h = 0;
                for(int i = 0; i < data.n_internal_frequencies; i++) {
                        casa::Vector<double> chan_freq_vector;
                        chan_freq_vector=chan_freq_col(i);
                        for(int j = 0; j < data.channels[i]; j++) {
                                fields[f].visibilities[h].freq = chan_freq_vector[j];
                                h++;
                        }
                }
        }

        for(int f=0; f<data.nfields; f++) {
                h = 0;
                fields[f].valid_frequencies = 0;
                for(int i = 0; i < data.n_internal_frequencies; i++) {
                        for(int j = 0; j < data.channels[i]; j++) {
                                if(fields[f].numVisibilitiesPerFreq[h] > 0) {
                                        fields[f].valid_frequencies++;
                                }
                                h++;
                        }
                }
        }


}

__host__ void MScopy(char const *in_dir, char const *in_dir_dest, int verbose_flag)
{
        string dir_origin = in_dir;
        string dir_dest = in_dir_dest;

        casa::Table tab_src(dir_origin);
        tab_src.deepCopy(dir_dest,casa::Table::New);
        if (verbose_flag) {
                cout << "Copied" << endl;
        }

}



__host__ void residualsToHost(Field *fields, freqData data, int num_gpus, int firstgpu)
{
        cout << "Saving residuals to host memory" << endl;
        if(num_gpus == 1) {
                for(int f=0; f<data.nfields; f++) {
                        for(int i=0; i<data.total_frequencies; i++) {
                                gpuErrchk(cudaMemcpy(fields[f].visibilities[i].Vm, fields[f].device_visibilities[i].Vm, sizeof(cufftComplex)*fields[f].numVisibilitiesPerFreq[i], cudaMemcpyDeviceToHost));
                                gpuErrchk(cudaMemcpy(fields[f].visibilities[i].weight, fields[f].device_visibilities[i].weight, sizeof(float)*fields[f].numVisibilitiesPerFreq[i], cudaMemcpyDeviceToHost));
                        }
                }
        }else{
                for(int f=0; f<data.nfields; f++) {
                        for(int i=0; i<data.total_frequencies; i++) {
                                cudaSetDevice((i%num_gpus) + firstgpu);
                                gpuErrchk(cudaMemcpy(fields[f].visibilities[i].Vm, fields[f].device_visibilities[i].Vm, sizeof(cufftComplex)*fields[f].numVisibilitiesPerFreq[i], cudaMemcpyDeviceToHost));
                                gpuErrchk(cudaMemcpy(fields[f].visibilities[i].weight, fields[f].device_visibilities[i].weight, sizeof(float)*fields[f].numVisibilitiesPerFreq[i], cudaMemcpyDeviceToHost));
                        }
                }
        }

        for(int f=0; f<data.nfields; f++) {
                for(int i=0; i<data.total_frequencies; i++) {
                        for(int j=0; j<fields[f].numVisibilitiesPerFreq[i]; j++) {
                                if(fields[f].visibilities[i].u[j]<0) {
                                        fields[f].visibilities[i].Vm[j].y *= -1;
                                }
                        }
                }
        }

}

__host__ void writeMS(char *infile, char *outfile, Field *fields, freqData data, float random_probability, int verbose_flag)
{
        MScopy(infile, outfile, verbose_flag);
        char* out_col = "DATA";
        string dir=outfile;
        string query;
        casa::Table main_tab(dir,casa::Table::Update);
        string column_name=out_col;

        if (main_tab.tableDesc().isColumn(column_name))
        {
                cout << "Column " << out_col << " already exists... skipping creation..." << endl;
        }else{
                cout << "Adding " << out_col << " to the main table..." << endl;
                main_tab.addColumn(casa::ArrayColumnDesc <casa::Complex>(column_name,"created by gpuvsim"));
                main_tab.flush();
        }

        if (column_name!="DATA")
        {
                query="UPDATE "+dir+" set "+column_name+"=DATA";
                cout << "Duplicating DATA column into " << out_col << endl;
                casa::tableCommand(query);
        }

        casa::TableRow row(main_tab, casa::stringToVector(column_name+",FLAG,FIELD_ID,WEIGHT,FLAG_ROW,DATA_DESC_ID"));
        casa::Vector<casa::Bool> auxbool;
        casa::Vector<float> weights;
        bool flag;
        int spw, field, h = 0, g = 0;
        for(int f=0; f<data.nfields; f++) {
                g=0;
                for(int i=0; i < data.n_internal_frequencies; i++) {
                        for(int j=0; j < data.channels[i]; j++) {
                                for (int k=0; k < data.nsamples; k++) {
                                        const casa::TableRecord &values = row.get(k);
                                        flag = values.asBool("FLAG_ROW");
                                        spw = values.asInt("DATA_DESC_ID");
                                        field = values.asInt("FIELD_ID");
                                        casa::Array<casa::Bool> flagCol = values.asArrayBool("FLAG");
                                        casa::Array<casa::Complex> dataCol = values.asArrayComplex(column_name);
                                        weights=values.asArrayFloat("WEIGHT");
                                        if(field == f && spw == i && flag == false) {
                                                for (int sto=0; sto< data.nstokes; sto++) {
                                                        auxbool = flagCol[j][sto];
                                                        if(auxbool[0] == false && weights[sto] > 0.0) {
                                                                dataCol[j][sto] = casa::Complex(fields[f].visibilities[g].Vo[h].x - fields[f].visibilities[g].Vm[h].x, fields[f].visibilities[g].Vo[h].y - fields[f].visibilities[g].Vm[h].y);
                                                                weights[sto] = fields[f].visibilities[g].weight[h];
                                                                h++;
                                                        }
                                                }
                                                row.put(k);
                                        }else continue;
                                }
                                h=0;
                                g++;
                        }
                }
        }
        main_tab.flush();

}

__host__ void writeMSSIM(char *infile, char *outfile, Field *fields, freqData data, int verbose_flag)
{
        MScopy(infile, outfile, verbose_flag);
        char* out_col = "DATA";
        string dir=outfile;
        string query;
        casa::Table main_tab(dir,casa::Table::Update);
        string column_name=out_col;

        if (main_tab.tableDesc().isColumn(column_name))
        {
                cout << "Column " << out_col << " already exists... skipping creation..." << endl;
        }else{
                cout << "Adding " << out_col << " to the main table..." << endl;
                main_tab.addColumn(casa::ArrayColumnDesc <casa::Complex>(column_name,"created by gpuvsim"));
                main_tab.flush();
        }

        if (column_name!="DATA")
        {
                query="UPDATE "+dir+" set "+column_name+"=DATA";
              cout << "Duplicating DATA column into " << out_col << endl;
                casa::tableCommand(query);
        }

        casa::TableRow row(main_tab, casa::stringToVector(column_name+",FLAG,FIELD_ID,WEIGHT,FLAG_ROW,DATA_DESC_ID"));
        casa::Vector<casa::Bool> auxbool;
        casa::Vector<float> weights;
        bool flag;
        int spw, field, h = 0, g = 0;
        for(int f=0; f<data.nfields; f++) {
                g=0;
                for(int i=0; i < data.n_internal_frequencies; i++) {
                        for(int j=0; j < data.channels[i]; j++) {
                                for (int k=0; k < data.nsamples; k++) {
                                        const casa::TableRecord &values = row.get(k);
                                        flag = values.asBool("FLAG_ROW");
                                        spw = values.asInt("DATA_DESC_ID");
                                        field = values.asInt("FIELD_ID");
                                        casa::Array<casa::Bool> flagCol = values.asArrayBool("FLAG");
                                        casa::Array<casa::Complex> dataCol = values.asArrayComplex(column_name);
                                        weights=values.asArrayFloat("WEIGHT");
                                        if(field == f && spw == i && flag == false) {
                                                for (int sto=0; sto< data.nstokes; sto++) {
                                                        auxbool = flagCol[j][sto];
                                                        if(auxbool[0] == false && weights[sto] > 0.0) {
                                                                dataCol[j][sto] = casa::Complex(fields[f].visibilities[g].Vm[h].x, fields[f].visibilities[g].Vm[h].y);
                                                                h++;
                                                        }
                                                }
                                                row.put(k);
                                        }else continue;
                                }
                                h=0;
                                g++;
                        }
                }
        }
        main_tab.flush();

}

__host__ void writeMSSIMMC(char *infile, char *outfile, Field *fields, freqData data, int verbose_flag)
{
        MScopy(infile, outfile, verbose_flag);
        char* out_col = "DATA";
        string dir=outfile;
        string query;
        casa::Table main_tab(dir,casa::Table::Update);
        string column_name=out_col;

        if (main_tab.tableDesc().isColumn(column_name))
        {
                cout << "Column " << out_col <<  " already exists... skipping creation..." << endl;
        }else{
                cout << "Adding " << out_col << " to the main table..." << endl;
                main_tab.addColumn(casa::ArrayColumnDesc <casa::Complex>(column_name,"created by gpuvsim"));
                main_tab.flush();
        }

        if (column_name!="DATA")
        {
                query="UPDATE "+dir+" set "+column_name+"=DATA";
                cout << "Duplicating DATA column into " << out_col << endl;
                casa::tableCommand(query);
        }

        casa::TableRow row(main_tab, casa::stringToVector(column_name+",FLAG,FIELD_ID,WEIGHT,FLAG_ROW,DATA_DESC_ID"));
        casa::Vector<casa::Bool> auxbool;
        casa::Vector<float> weights;
        bool flag;
        int spw, field, h = 0, g = 0;
        float real_n, imag_n;
        SelectStream(0);
        PutSeed(-1);

        for(int f=0; f<data.nfields; f++) {
                g=0;
                for(int i=0; i < data.n_internal_frequencies; i++) {
                        for(int j=0; j < data.channels[i]; j++) {
                                for (int k=0; k < data.nsamples; k++) {
                                        const casa::TableRecord &values = row.get(k);
                                        flag = values.asBool("FLAG_ROW");
                                        spw = values.asInt("DATA_DESC_ID");
                                        field = values.asInt("FIELD_ID");
                                        casa::Array<casa::Bool> flagCol = values.asArrayBool("FLAG");
                                        casa::Array<casa::Complex> dataCol = values.asArrayComplex(column_name);
                                        weights=values.asArrayFloat("WEIGHT");
                                        if(field == f && spw == i && flag == false) {
                                                for (int sto=0; sto< data.nstokes; sto++) {
                                                        auxbool = flagCol[j][sto];
                                                        if(auxbool[0] == false && weights[sto] > 0.0) {
                                                                real_n = Normal(0.0, 1.0);
                                                                imag_n = Normal(0.0, 1.0);
                                                                dataCol[j][sto] = casa::Complex(fields[f].visibilities[g].Vm[h].x + real_n * (1/sqrt(weights[sto])), fields[f].visibilities[g].Vm[h].y + imag_n * (1/sqrt(weights[sto])));
                                                                h++;
                                                        }
                                                }
                                                row.put(k);
                                        }else continue;
                                }
                                h=0;
                                g++;
                        }
                }
        }
        main_tab.flush();

}

__host__ void writeMSSIMSubsampled(char *infile, char *outfile, Field *fields, freqData data, float random_probability, int verbose_flag)
{
        MScopy(infile, outfile, verbose_flag);
        char* out_col = "DATA";
        string dir=outfile;
        string query;
        casa::Table main_tab(dir,casa::Table::Update);
        string column_name=out_col;

        if (main_tab.tableDesc().isColumn(column_name))
        {
                cout << "Column " << out_col << " already exists... skipping creation..." << endl;
        }else{
                cout << "Adding " << out_col << " to the main table..." << endl;
                main_tab.addColumn(casa::ArrayColumnDesc <casa::Complex>(column_name,"created by gpuvsim"));
                main_tab.flush();
        }

        if (column_name!="DATA")
        {
                query="UPDATE "+dir+" set "+column_name+"=DATA";
                cout << "Duplicating DATA column into " << out_col << endl;
                casa::tableCommand(query);
        }

        casa::TableRow row(main_tab, casa::stringToVector(column_name+",FLAG,FIELD_ID,WEIGHT,FLAG_ROW,DATA_DESC_ID"));
        casa::Vector<casa::Bool> auxbool;
        casa::Vector<float> weights;
        bool flag;
        int spw, field, h = 0, g = 0;
        float u;
        SelectStream(0);
        PutSeed(-1);

        for(int f=0; f<data.nfields; f++) {
                g=0;
                for(int i=0; i < data.n_internal_frequencies; i++) {
                        for(int j=0; j < data.channels[i]; j++) {
                                for (int k=0; k < data.nsamples; k++) {
                                        const casa::TableRecord &values = row.get(k);
                                        flag = values.asBool("FLAG_ROW");
                                        spw = values.asInt("DATA_DESC_ID");
                                        field = values.asInt("FIELD_ID");
                                        casa::Array<casa::Bool> flagCol = values.asArrayBool("FLAG");
                                        casa::Array<casa::Complex> dataCol = values.asArrayComplex(column_name);
                                        weights=values.asArrayFloat("WEIGHT");
                                        if(field == f && spw == i && flag == false) {
                                                for (int sto=0; sto< data.nstokes; sto++) {
                                                        auxbool = flagCol[j][sto];
                                                        if(auxbool[0] == false && weights[sto] > 0.0) {
                                                                u = Random();
                                                                if(u<random_probability) {
                                                                        dataCol[j][sto] = casa::Complex(fields[f].visibilities[g].Vm[h].x, fields[f].visibilities[g].Vm[h].y);
                                                                }else{
                                                                        dataCol[j][sto] = casa::Complex(fields[f].visibilities[g].Vm[h].x, fields[f].visibilities[g].Vm[h].y);
                                                                        weights[sto] = 0.0;
                                                                }
                                                                h++;
                                                        }
                                                }
                                                row.put(k);
                                        }else continue;
                                }
                                h=0;
                                g++;
                        }
                }
        }
        main_tab.flush();

}


__host__ void writeMSSIMSubsampledMC(char *infile, char *outfile, Field *fields, freqData data, float random_probability, int verbose_flag)
{
        MScopy(infile, outfile, verbose_flag);
        char* out_col = "DATA";
        string dir=outfile;
        string query;
        casa::Table main_tab(dir,casa::Table::Update);
        string column_name=out_col;

        if (main_tab.tableDesc().isColumn(column_name))
        {
                cout << "Column " << out_col  << " already exists... skipping creation..." << endl;
        }else{
                cout << "Adding " << out_col << " to the main table..." << endl;
                main_tab.addColumn(casa::ArrayColumnDesc <casa::Complex>(column_name,"created by gpuvsim"));
                main_tab.flush();
        }

        if (column_name!="DATA")
        {
                query="UPDATE "+dir+" set "+column_name+"=DATA";
                cout << "Duplicating DATA column into " << out_col << endl;
                casa::tableCommand(query);
        }

        casa::TableRow row(main_tab, casa::stringToVector(column_name+",FLAG,FIELD_ID,WEIGHT,FLAG_ROW,DATA_DESC_ID"));
        casa::Vector<casa::Bool> auxbool;
        casa::Vector<float> weights;
        bool flag;
        int spw, field, h = 0, g = 0;
        float real_n, imag_n;
        float u;
        SelectStream(0);
        PutSeed(-1);

        for(int f=0; f<data.nfields; f++) {
                g=0;
                for(int i=0; i < data.n_internal_frequencies; i++) {
                        for(int j=0; j < data.channels[i]; j++) {
                                for (int k=0; k < data.nsamples; k++) {
                                        const casa::TableRecord &values = row.get(k);
                                        flag = values.asBool("FLAG_ROW");
                                        spw = values.asInt("DATA_DESC_ID");
                                        field = values.asInt("FIELD_ID");
                                        casa::Array<casa::Bool> flagCol = values.asArrayBool("FLAG");
                                        casa::Array<casa::Complex> dataCol = values.asArrayComplex(column_name);
                                        weights=values.asArrayFloat("WEIGHT");
                                        if(field == f && spw == i && flag == false) {
                                                for (int sto=0; sto< data.nstokes; sto++) {
                                                        auxbool = flagCol[j][sto];
                                                        if(auxbool[0] == false && weights[sto] > 0.0) {
                                                                u = Random();
                                                                if(u<random_probability) {
                                                                        real_n = Normal(0.0, 1.0);
                                                                        imag_n = Normal(0.0, 1.0);
                                                                        dataCol[j][sto] = casa::Complex(fields[f].visibilities[g].Vm[h].x + real_n * (1/sqrt(weights[sto])), fields[f].visibilities[g].Vm[h].y + imag_n * (1/sqrt(weights[sto])));
                                                                }else{
                                                                        dataCol[j][sto] = casa::Complex(fields[f].visibilities[g].Vm[h].x, fields[f].visibilities[g].Vm[h].y);
                                                                        weights[sto] = 0.0;
                                                                }
                                                                h++;
                                                        }
                                                }
                                                row.put(k);
                                        }else continue;
                                }
                                h=0;
                                g++;
                        }
                }
        }
        main_tab.flush();

}

__host__ void OFITS(float *I, fitsfile *canvas, char *path, std::string name_image, char *units, int iteration, int index, float fg_scale, long M, long N)
{
        fitsfile *fpointer;
        int status = 0;
        long fpixel = 1;
        long elements = M*N;
        size_t needed;
        long naxes[2]={M,N};
        long naxis = 2;
        std::stringstream full_name {};
        /*
        needed = snprintf(NULL, 0, "!%s%s", path, name_image) + 1;
        full_name = (char*)malloc(needed*sizeof(char));
        snprintf(full_name, needed*sizeof(char), "!%s%s", path, name_image);*/

        full_name << "!" << path << name_image;

        fits_create_file(&fpointer, full_name.str(), &status);
        if(status) {
                fits_report_error(stderr, status); /* print error message */
                exit(-1);
        }

        fits_copy_header(canvas, fpointer, &status);
        if (status) {
                fits_report_error(stderr, status); /* print error message */
                exit(-1);
        }

        fits_update_key(fpointer, TSTRING, "BUNIT", units, "Unit of measurement", &status);
        fits_update_key(fpointer, TINT, "NITER", &iteration, "Number of iteration in gpuvmem software", &status);

        float *host_IFITS = (float*)malloc(M*N*sizeof(float));

        //unsigned int offset = M*N*index*sizeof(float);
        int offset = M*N*index;
        gpuErrchk(cudaMemcpy(host_IFITS, &I[offset], sizeof(float)*M*N, cudaMemcpyDeviceToHost));

        float *image2D = (float*) malloc(M*N*sizeof(float));

        int x = M-1;
        int y = N-1;
        for(int i=0; i < M; i++) {
                for(int j=0; j < N; j++) {
                        if(fg_scale != 0.0)
                                image2D[N*(y-i)+(x-j)] = host_IFITS[N*i+j] * fg_scale;
                        else
                                image2D[N*(y-i)+(x-j)] = host_IFITS[N*i+j];
                }
        }

        fits_write_img(fpointer, TFLOAT, fpixel, elements, image2D, &status);
        if (status) {
                fits_report_error(stderr, status); /* print error message */
                exit(-1);
        }
        fits_close_file(fpointer, &status);
        if (status) {
                fits_report_error(stderr, status); /* print error message */
                exit(-1);
        }

        free(host_IFITS);
        free(image2D);

}

__host__ void float2toImage(float *I, fitsfile *canvas, char *out_image, char*mempath, int iteration, float fg_scale, long M, long N, int option)
{
        fitsfile *fpointerI_nu_0, *fpointeralpha, *fpointer;
        int statusI_nu_0 = 0, statusalpha = 0;
        long fpixel = 1;
        long elements = M*N;
        char *Inu_0_name;
        char *alphaname;
        size_t needed_I_nu_0;
        size_t needed_alpha;
        long naxes[2]={M,N};
        long naxis = 2;
        char *alphaunit = "";
        char *I_unit = "JY/PIXEL";

        float *host_2Iout = (float*)malloc(M*N*sizeof(float)*2);

        gpuErrchk(cudaMemcpy(host_2Iout, I, sizeof(float)*M*N*2, cudaMemcpyDeviceToHost));

        float *host_alpha = (float*)malloc(M*N*sizeof(float));
        float *host_I_nu_0 = (float*)malloc(M*N*sizeof(float));

        switch(option) {
        case 0:
                needed_alpha = snprintf(NULL, 0, "!%s_alpha.fits", out_image) + 1;
                alphaname = (char*)malloc(needed_alpha*sizeof(char));
                snprintf(alphaname, needed_alpha*sizeof(char), "!%s_alpha.fits", out_image);
                break;
        case 1:
                needed_alpha = snprintf(NULL, 0, "!%salpha_%d.fits", mempath, iteration) + 1;
                alphaname = (char*)malloc(needed_alpha*sizeof(char));
                snprintf(alphaname, needed_alpha*sizeof(char), "!%salpha_%d.fits", mempath, iteration);
                break;
        case 2:
                needed_alpha = snprintf(NULL, 0, "!%salpha_error.fits", out_image) + 1;
                alphaname = (char*)malloc(needed_alpha*sizeof(char));
                snprintf(alphaname, needed_alpha*sizeof(char), "!%salpha_error.fits", out_image);
                break;
        case -1:
                break;
        default:
                cout << "Invalid case to FITS" << endl;
                exit(-1);
        }

        switch(option) {
        case 0:
                needed_I_nu_0 = snprintf(NULL, 0, "!%s_I_nu_0.fits", out_image) + 1;
                Inu_0_name = (char*)malloc(needed_I_nu_0*sizeof(char));
                snprintf(Inu_0_name, needed_I_nu_0*sizeof(char), "!%s_I_nu_0.fits", out_image);
                break;
        case 1:
                needed_I_nu_0 = snprintf(NULL, 0, "!%sI_nu_0_%d.fits", mempath, iteration) + 1;
                Inu_0_name = (char*)malloc(needed_I_nu_0*sizeof(char));
                snprintf(Inu_0_name, needed_I_nu_0*sizeof(char), "!%sI_nu_0_%d.fits", mempath, iteration);
                break;
        case 2:
                needed_I_nu_0 = snprintf(NULL, 0, "!%s_I_nu_0_error.fits", out_image) + 1;
                Inu_0_name = (char*)malloc(needed_I_nu_0*sizeof(char));
                snprintf(Inu_0_name, needed_I_nu_0*sizeof(char), "!%s_I_nu_0_error.fits", out_image);
        case -1:
                break;
        default:
                cout << "Invalid case to FITS" << endl;
                exit(-1);
        }


        fits_create_file(&fpointerI_nu_0, Inu_0_name, &statusI_nu_0);
        fits_create_file(&fpointeralpha, alphaname, &statusalpha);

        if (statusI_nu_0 || statusalpha) {
                fits_report_error(stderr, statusI_nu_0);
                fits_report_error(stderr, statusalpha);
                exit(-1); /* print error message */
        }

        fits_copy_header(canvas, fpointerI_nu_0, &statusI_nu_0);
        fits_copy_header(canvas, fpointeralpha, &statusalpha);

        if (statusI_nu_0 || statusalpha) {
                fits_report_error(stderr, statusI_nu_0);
                fits_report_error(stderr, statusalpha);
                exit(-1); /* print error message */
        }

        fits_update_key(fpointerI_nu_0, TSTRING, "BUNIT", I_unit, "Unit of measurement", &statusI_nu_0);
        fits_update_key(fpointeralpha, TSTRING, "BUNIT", alphaunit, "Unit of measurement", &statusalpha);

        int x = M-1;
        int y = N-1;
        for(int i=0; i < M; i++) {
                for(int j=0; j < N; j++) {
                        host_I_nu_0[N*(y-i)+(x-j)] = host_2Iout[N*i+j];
                        host_alpha[N*(y-i)+(x-j)] = host_2Iout[N*M+N*i+j];
                }
        }

        fits_write_img(fpointerI_nu_0, TFLOAT, fpixel, elements, host_I_nu_0, &statusI_nu_0);
        fits_write_img(fpointeralpha, TFLOAT, fpixel, elements, host_alpha, &statusalpha);

        if (statusI_nu_0 || statusalpha) {
                fits_report_error(stderr, statusI_nu_0);
                fits_report_error(stderr, statusalpha);
                exit(-1); /* print error message */
        }
        fits_close_file(fpointerI_nu_0, &statusI_nu_0);
        fits_close_file(fpointeralpha, &statusalpha);

        if (statusI_nu_0 || statusalpha) {
                fits_report_error(stderr, statusI_nu_0);
                fits_report_error(stderr, statusalpha);
                exit(-1); /* print error message */
        }

        free(host_I_nu_0);
        free(host_alpha);

        free(host_2Iout);

        free(alphaname);
        free(Inu_0_name);


}

__host__ void float3toImage(float3 *I, fitsfile *canvas, char *out_image, char*mempath, int iteration, long M, long N, int option)
{
        fitsfile *fpointerT, *fpointertau, *fpointerbeta, *fpointer;
        int statusT = 0, statustau = 0, statusbeta = 0;
        long fpixel = 1;
        long elements = M*N;
        char *Tname;
        char *tauname;
        char *betaname;
        size_t needed_T;
        size_t needed_tau;
        size_t needed_beta;
        long naxes[2]={M,N};
        long naxis = 2;
        char *Tunit = "K";
        char *tauunit = "";
        char *betaunit = "";

        float3 *host_3Iout = (float3*)malloc(M*N*sizeof(float3));

        gpuErrchk(cudaMemcpy2D(host_3Iout, sizeof(float3), I, sizeof(float3), sizeof(float3), M*N, cudaMemcpyDeviceToHost));

        float *host_T = (float*)malloc(M*N*sizeof(float));
        float *host_tau = (float*)malloc(M*N*sizeof(float));
        float *host_beta = (float*)malloc(M*N*sizeof(float));

        switch(option) {
        case 0:
                needed_T = snprintf(NULL, 0, "!%s_T.fits", out_image) + 1;
                Tname = (char*)malloc(needed_T*sizeof(char));
                snprintf(Tname, needed_T*sizeof(char), "!%s_T.fits", out_image);
                break;
        case 1:
                needed_T = snprintf(NULL, 0, "!%sT_%d.fits", mempath, iteration) + 1;
                Tname = (char*)malloc(needed_T*sizeof(char));
                snprintf(Tname, needed_T*sizeof(char), "!%sT_%d.fits", mempath, iteration);
                break;
        case -1:
                break;
        default:
                cout << "Invalid case to FITS" << endl;
                exit(-1);
        }

        switch(option) {
        case 0:
                needed_tau = snprintf(NULL, 0, "!%s_tau_0.fits", out_image) + 1;
                tauname = (char*)malloc(needed_tau*sizeof(char));
                snprintf(tauname, needed_tau*sizeof(char), "!%s_tau_0.fits", out_image);
                break;
        case 1:
                needed_tau = snprintf(NULL, 0, "!%stau_0_%d.fits", mempath, iteration) + 1;
                tauname = (char*)malloc(needed_tau*sizeof(char));
                snprintf(tauname, needed_tau*sizeof(char), "!%stau_0_%d.fits", mempath, iteration);
                break;
        case -1:
                break;
        default:
                cout << "Invalid case to FITS" << endl;
                exit(-1);
        }

        switch(option) {
        case 0:
                needed_beta = snprintf(NULL, 0, "!%s_beta.fits", out_image) + 1;
                betaname = (char*)malloc(needed_beta*sizeof(char));
                snprintf(betaname, needed_beta*sizeof(char), "!%s_beta.fits", out_image);
                break;
        case 1:
                needed_beta = snprintf(NULL, 0, "!%sbeta_%d.fits", mempath, iteration) + 1;
                betaname = (char*)malloc(needed_beta*sizeof(char));
                snprintf(betaname, needed_beta*sizeof(char), "!%sbeta_%d.fits", mempath, iteration);
                break;
        case -1:
                break;
        default:
                cout << "Invalid case to FITS" << endl;
                exit(-1);
        }

        fits_create_file(&fpointerT, Tname, &statusT);
        fits_create_file(&fpointertau, tauname, &statustau);
        fits_create_file(&fpointerbeta, betaname, &statusbeta);

        if (statusT || statustau || statusbeta) {
                fits_report_error(stderr, statusT);
                fits_report_error(stderr, statustau);
                fits_report_error(stderr, statusbeta);
                exit(-1);
        }

        fits_copy_header(canvas, fpointerT, &statusT);
        fits_copy_header(canvas, fpointertau, &statustau);
        fits_copy_header(canvas, fpointerbeta, &statusbeta);

        if (statusT || statustau || statusbeta) {
                fits_report_error(stderr, statusT);
                fits_report_error(stderr, statustau);
                fits_report_error(stderr, statusbeta);
                exit(-1);
        }

        fits_update_key(fpointerT, TSTRING, "BUNIT", Tunit, "Unit of measurement", &statusT);
        fits_update_key(fpointertau, TSTRING, "BUNIT", tauunit, "Unit of measurement", &statustau);
        fits_update_key(fpointerbeta, TSTRING, "BUNIT", betaunit, "Unit of measurement", &statusbeta);

        int x = M-1;
        int y = N-1;
        for(int i=0; i < M; i++) {
                for(int j=0; j < N; j++) {
                        host_T[N*(y-i)+(x-j)] = host_3Iout[N*i+j].x;
                        host_tau[N*(y-i)+(x-j)] = host_3Iout[N*i+j].y;
                        host_beta[N*(y-i)+(x-j)] = host_3Iout[N*i+j].z;
                }
        }

        fits_write_img(fpointerT, TFLOAT, fpixel, elements, host_T, &statusT);
        fits_write_img(fpointertau, TFLOAT, fpixel, elements, host_tau, &statustau);
        fits_write_img(fpointerbeta, TFLOAT, fpixel, elements, host_beta, &statusbeta);
        if (statusT || statustau || statusbeta) {
                fits_report_error(stderr, statusT);
                fits_report_error(stderr, statustau);
                fits_report_error(stderr, statusbeta);
                exit(-1);
        }
        fits_close_file(fpointerT, &statusT);
        fits_close_file(fpointertau, &statustau);
        fits_close_file(fpointerbeta, &statusbeta);
        if (statusT || statustau || statusbeta) {
                fits_report_error(stderr, statusT);
                fits_report_error(stderr, statustau);
                fits_report_error(stderr, statusbeta);
                exit(-1);
        }

        free(host_T);
        free(host_tau);
        free(host_beta);
        free(host_3Iout);

        free(betaname);
        free(tauname);
        free(Tname);

}

__host__ void closeCanvas(fitsfile *canvas)
{
        int status = 0;
        fits_close_file(canvas, &status);
        if(status) {
                fits_report_error(stderr, status);
                exit(-1);
        }
}
