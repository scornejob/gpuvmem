#include "ioms.cuh"

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
