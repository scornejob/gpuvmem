#include "error.cuh"

extern long N, M;

void SecondDerivateError::calculateErrorImage(Image *I, Visibilities *v)
{
  /* params must contain image and visibilities,
  is recomended tu make a strunct for it in framework.cuh.
  Image object has a float * for the error images, but it is not initialized.
  */
};

namespace {
  Error* CreateSecondDerivateError()
  {
    return new SecondDerivateError;
  }
  const int SecondDerivateErrorID = 0;
  const bool RegisteredSecondDerivateError = Singleton<ErrorFactory>::Instance().RegisterError(SecondDerivateErrorID, CreateSecondDerivateError);
};
