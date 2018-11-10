#include "ioms.cuh"

extern long M,N;
extern float fg_scale;
extern char* mempath;
extern fitsfile *mod_in;
extern int iter;

freqData IoMS::IocountVisibilities(char * MS_name, Field *&fields)
{
        return countVisibilities(MS_name, fields);
};
canvasVariables IoMS::IoreadCanvas(char *canvas_name, fitsfile *&canvas, float b_noise_aux, int status_canvas, int verbose_flag)
{
        return readCanvas(canvas_name, canvas, b_noise_aux, status_canvas, verbose_flag);
};
void IoMS::IoreadMSMCNoise(char *MS_name, Field *fields, freqData data)
{
        readMSMCNoise(MS_name, fields, data);
};
void IoMS::IoreadSubsampledMS(char *MS_name, Field *fields, freqData data, float random_probability)
{
        readSubsampledMS(MS_name, fields, data, random_probability);
};
void IoMS::IoreadMCNoiseSubsampledMS(char *MS_name, Field *fields, freqData data, float random_probability)
{
        readMCNoiseSubsampledMS(MS_name, fields, data, random_probability);
};
void IoMS::IoreadMS(char *MS_name, Field *fields, freqData data)
{
        readMS(MS_name, fields, data);
};
void IoMS::IowriteMS(char *infile, char *outfile, Field *fields, freqData data, float random_probability, int verbose_flag)
{
        writeMS(infile, outfile, fields, data, random_probability, verbose_flag);
};

void IoMS::IoPrintImage(float *I, fitsfile *canvas, char *path, char *name_image, char *units, int iteration, int index, float fg_scale, long M, long N)
{
        OFITS(I, canvas, path, name_image, units, iteration, index, fg_scale, M, N);
}

void IoMS::IoPrintImageIteration(float *I, fitsfile *canvas, char *path, char *name_image, char *units, int iteration, int index, float fg_scale, long M, long N)
{
        /*
        size_t needed;
        char *full_name;

        needed = snprintf(NULL, 0, "%s_%d.fits", name_image, iteration) + 1;
        full_name = (char*)malloc(needed*sizeof(char));
        snprintf(full_name, needed*sizeof(char), "%s_%d.fits", name_image, iteration);

        OFITS(I, canvas, path, full_name, units, iteration, index, fg_scale, M, N);
        free(full_name);*/
        std::stringstream full_name;
        std::string format (".fits");
        full_name << name_image << "_" << iteration << format;
	char* tmp_full_name = new char[full_name.str().length() + 1];
        std::copy(full_name.str().c_str(),full_name.str().c_str() + full_name.str().length()+1,tmp_full_name);
        OFITS(I, canvas, path, tmp_full_name, units, iteration, index, fg_scale, M, N);
	delete[] tmp_full_name;
}

void IoMS::IoPrintMEMImageIteration(float *I, char *name_image, char *units, int index)
{
	/*
        size_t needed;
        char *full_name;

        needed = snprintf(NULL, 0, "%s_%d.fits", name_image, iter) + 1;
        full_name = (char*)malloc(needed*sizeof(char));
        snprintf(full_name, needed*sizeof(char), "%s_%d.fits", name_image, iter);

        OFITS(I, mod_in, mempath, full_name, units, iter, index, fg_scale, M, N);
        free(full_name);*/
	std::stringstream full_name;
        std::string format (".fits");
        full_name << name_image << "_" << iteration << format;
	char* tmp_full_name = new char[full_name.str().length() + 1];
        std::copy(full_name.str().c_str(),full_name.str().c_str() + full_name.str().length()+1,tmp_full_name);
	OFITS(I, mod_in, mempath, tmp_full_name, units, iter, index, fg_scale, M, N);
	delete[] tmp_full_name;
}

void IoMS::IocloseCanvas(fitsfile *canvas)
{
        closeCanvas(canvas);
};

namespace {
Io* CreateIoMS()
{
        return new IoMS;
}
const int IoMSId = 0;
const bool RegisteredIoMS = Singleton<IoFactory>::Instance().RegisterIo(IoMSId, CreateIoMS);
};
