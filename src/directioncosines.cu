#include "directioncosines.cuh"
/*--------------------------------------------------------------------
 * Convert (ra,dec) to direction cosines (l,m) relative to
 * phase-tracking center (ra0, dec0). All in radians.
 * Reference: Synthesis Imaging in Radio Astronomy II, p.388.
 *--------------------------------------------------------------------*/
__host__ void direccos(float ra, float dec, float ra0, float dec0, float* l, float* m)
{
  float delta_ra = ra - ra0;
  float cosdec = cos(dec);
  *l = cosdec * sin(delta_ra);
  *m = sin(dec) * cos(dec0) - cosdec * sin(dec0) * cos(delta_ra);
}
