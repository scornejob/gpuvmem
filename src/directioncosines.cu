#include "directioncosines.cuh"
/**
* Convert (ra,dec) to direction cosines (l,m) relative to
* phase-tracking center (ra0, dec0). All in radians.
* Reference: Synthesis Imaging in Radio Astronomy II, p.388.
 */
__host__ void direccos(double ra, double dec, double ra0, double dec0, double* l, double* m)
{
  double delta_ra = ra - ra0;
  double cosdec = cos(dec);
  *l = cosdec * sin(delta_ra);
  *m = sin(dec) * cos(dec0) - cosdec * sin(dec0) * cos(delta_ra);
}
