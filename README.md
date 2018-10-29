# GPUVMEM

# Papers and documentation

- https://doi.org/10.1016/j.ascom.2017.11.003


#Citing

If you use GPUVMEM for your research please do not forget to cite Cárcamo et al.

```
@article{CARCAMO201816,
title = "Multi-GPU maximum entropy image synthesis for radio astronomy",
journal = "Astronomy and Computing",
volume = "22",
pages = "16 - 27",
year = "2018",
issn = "2213-1337",
doi = "https://doi.org/10.1016/j.ascom.2017.11.003",
url = "http://www.sciencedirect.com/science/article/pii/S2213133717300094",
author = "M. Cárcamo and P.E. Román and S. Casassus and V. Moral and F.R. Rannou",
keywords = "Maximum entropy, GPU, ALMA, Inverse problem, Radio interferometry, Image synthesis"
}
```

# Installation

1. Download or clone gpuvmem.

2. To compile GPUVMEM you will need:

- casacore (https://github.com/casacore/casacore, please make sure you have installed the github version, Ubuntu package doesn't work well since doesn't have the `put()` function). Additionally, if you are using the last version of casacore, please compile using the flag NEWCASA=1.
- CUDA >= 9
- OpenMP

# Compiling
```
cd gpuvmem
./configure
make
```

If you are using the last version of casacore
 ```
cd gpuvmem
./configure
make NEWCASA=1
```

# Usage

Create your canvas or mod_in_0.fits with the image data in the header, typically we use a FITS CASA CLEAN dirty image of the desired object.

Create your input.dat file with the following data and change the parameters as you need:

```
noise_cut	100.5
ftol		1.0e-12
random_probability 0.0
t_telescope        2
```
# t_telescope references the following telescopes:

1. CBI2
2. ALMA
3. ATCA
4. VLA
5. SZA
6. CBI


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
-e  --eta             Variable that controls the minimum image value (Default eta = -1.0)
-f  --file            Output file where final objective function values are saved (Optional)
-M  --multigpu        Number of GPUs to use multiGPU image synthesis (Default OFF => 0)
-s  --select          If multigpu option is OFF, then select the GPU ID of the GPU you will work on. (Default = 0)
-t  --iterations      Number of iterations for optimization (Default = 50)
-z  --initial-cond    Initial conditions for image/s
-Z  --penalizators    Penalizators for Fi/s in Objective Function
    --xcorr           Run gpuvmem with cross-correlation
    --nopositivity    Run gpuvmem using chi2 with no positivity restriction
    --apply-noise     Apply random gaussian noise to visibilities
    --clipping        Clips the image to positive values
    --print-images    Prints images per iteration
    --verbose         Shows information through all the execution
```
# Framework usage

The normal flow of the program starts by creating a synthesizer, creating an optimizer, creating an objective function, and adding the Fi to the objective function

All the objects must be created by their respective factory

The number of available images is determined by the -z command

Gridding can be applied both as a filter and as an input parameter

All the specializations of objects are listed in the enums

All filters can only be applied before using Synthesizer->setDevice()

The configuration of a Fi has as parameters, the index of its penalty factor (-Z), the index of the image from where the data will be calculated, and finally the index of the image where the results of the calculations will be applied.

# IMPORTANT

Remember to create the mem folder to save the FITS images

# Fixes

We have fixed the Makefile and now you can compile gpuvmem using the new version of casacore.

# TO RESTORE YOUR IMAGE PLEASE SEE CARCAMO ET AL. 2018 FOR MORE INFORMATION

Restoring usage:

`casapy --log2term --nogui -c restore_continuum_ms.py residual_folder.ms mem_model.fits restored_output`

# CONTRIBUTORS

- Miguel Cárcamo - Universidad de Santiago de Chile - miguel.carcamo@usach.cl
- Nicolás Muñoz - Universidad de Santiago de Chile
- Fernando Rannou - Universidad de Santiago de Chile
- Pablo Román - Universidad de Santiago de Chile
- Simón Casassus - Universidad de Chile -
- Axel Osses - Universidad de Chile
- Victor Moral - Universidad de Chile

# CONTRIBUTION AND BUG REPORTS

**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

**Expected behavior**
A clear and concise description of what you expected to happen.

**Screenshots**
If applicable, add screenshots to help explain your problem.

**Desktop (please complete the following information):**
 - OS: [e.g. Ubuntu 16.04]
 - CUDA version [e.g. 9]
 - gpuvmem Version [e.g. 22]

**Additional context**
Add any other context about the problem here.

# FEATURE REQUEST

**Is your feature request related to a problem? Please describe.**
A clear and concise description of what the problem is. Ex. I'm always frustrated when [...]

**Describe the solution you'd like**
A clear and concise description of what you want to happen.

**Describe alternatives you've considered**
A clear and concise description of any alternative solutions or features you've considered.

**Additional context**
Add any other context or screenshots about the feature request here.
