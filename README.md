# GPUVMEM

#Libraries

To compile GPUVMEM you need:

- casacore
- CUDA
- OpenMP

#Compiling
```
cd gpuvmem
./configure
make
```
#Usage

Create your beam_0.fits and mod_in_0.fits with difmap.
Create your input.dat file with the following data and change the parameters if you want:

```
lambda_entropy  0.01
noise_cut	100.5
minpix_factor   1000.0
ftol		1.0e-12
random_probability 0.0
```

Use GPUVMEM:

Example: `./bin/gpuvmem options [ arguments ...]`
```
-h  --help            Shows this
-X --blockSizeX       Block X Size for Image (Needs to be pow of 2)
-Y --blockSizeY       Block Y Size for Image (Needs to be pow of 2)
-V --blockSizeV       Block Size for Visibilities (Needs to be pow of 2)
-i  --input           The name of the input file of visibilities(MS)
-o  --output          The name of the output file of residual visibilities(MS)
-O  --output-image    The name of the output image FITS file
-I  --inputdat        The name of the input file of parameters
-m  --modin           mod_in_0 FITS file
-b  --beam            beam_0 FITS file
-p  --path            MEM folder path to save FITS images. With last / included. (Example ./../mem/)
-M  --multigpu        Number of GPUs to use multiGPU image synthesis (Default OFF => 0)
-s  --select          If multigpu option is OFF, then select the GPU ID of the GPU you will work on. (Default = 0)
-t  --iterations      Number of iterations for optimization (Default = 50)
    --xcorr           Run gpuvmem with cross-correlation
    --verbose         Shows information through all the execution
    --nopositivity    Run gpuvmem using chi2 with no posititivy restriction
```
#IMPORTANT

Remember to create the mem folder to save the FITS images

#RESTORING YOUR IMAGE

Usage:

- RUN in the same folder:
`casapy --log2term --nogui -c restore_continuum_ms.py`

- CONFIG:
On the same file please edit first seven lines
```
residual_ms = "<MS_RESIDUAL_VISIBILITIES>"
model_fits = "<FITS_MODEL_IMAGE>"
restored = "<RESTORED_OUTPUT_FILENAME>"
pix_size="0.0084arcsec"
pix_num=2048
weight="briggs"
polarization="I"
```

#Contributors

- Miguel Cárcamo
- Fernando Rannou
- Pablo Román
- Simón Casassus
- Victor Moral
