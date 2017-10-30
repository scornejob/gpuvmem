# GPUVMEM

# Libraries

To compile GPUVMEM you need:

- casacore (https://github.com/casacore/casacore, please make sure you have installed the github version, Ubuntu package doesn't work well since doesn't have the `put()` function)
- CUDA
- OpenMP

# Compiling
```
cd gpuvmem
./configure
make
```
# Usage

Create your mod_in_0.fits with difmap or CASA.
Create your input.dat file with the following data and change the parameters if you want:

```
lambda_entropy  0.01
noise_cut	100.5
minpix  0.001
ftol		1.0e-12
random_probability 0.0
t_telescope        2
```
# t_telescope references the following telescopes:

1. CBI2
2. ALMA
3. CBI2 Test
4. ATCA
5. VLA
6. SZA


# Use GPUVMEM:

Example: `./bin/gpuvmem options [ arguments ...]`
```
-h  --help            Shows this
-X  --blockSizeX      Block X Size for Image (Needs to be pow of 2)
-Y  --blockSizeY      Block Y Size for Image (Needs to be pow of 2)
-V  --blockSizeV      Block Size for Visibilities (Needs to be pow of 2)
-i  --input           The name of the input file of visibilities(MS)
-o  --output          The name of the output file of residual visibilities(MS)
-O  --output-image    The name of the output image FITS file
-I  --inputdat        The name of the input file of parameters
-m  --modin           mod_in_0 FITS file
-x  --minpix          Minimum positive value of a pixel (Optional)
-n  --noise           Noise parameter (Optional)
-N  --noise-cut       Noise-cut Parameter (Optional)
-l  --lambda          Lambda Regularization Parameter (Optional)
-r  --randoms         Percentage of data used when random sampling (Default = 1.0, optional)
-p  --path            MEM folder path to save FITS images. With last / included. (Example ./../mem/)
-P  --prior           Prior used to regularize the solution (Default = 0 = Entropy)
-e  --eta              Variable that controls the minimum image value (Default eta = -1.0)
-f  --file            Output file where final objective function values are saved (Optional)
-M  --multigpu        Number of GPUs to use multiGPU image synthesis (Default OFF => 0)
-s  --select          If multigpu option is OFF, then select the GPU ID of the GPU you will work on. (Default = 0)
-t  --iterations      Number of iterations for optimization (Default = 50)
    --xcorr           Run gpuvmem with cross-correlation
    --nopositivity    Run gpuvmem using chi2 with no positivity restriction
    --apply-noise     Apply random gaussian noise to visibilities
    --clipping        Clips the image to positive values
    --print-images    Prints images per iteration
    --verbose         Shows information through all the execution
```


# PRIORS

0. Entropy
1. Quadratic variation
2. Total Variation

# IMPORTANT

Remember to create the mem folder to save the FITS images

# RESTORING YOUR IMAGE

Usage:

`casapy --log2term --nogui -c restore_continuum_ms.py residual_folder.ms mem_model.fits restored_output`

# Contributors

- Miguel Cárcamo
- Fernando Rannou
- Pablo Román
- Simón Casassus
- Victor Moral
