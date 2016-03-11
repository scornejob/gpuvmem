# GPUVMEM

#Libraries

To compile GPUVMEM you need:

- sqlite3
- CUDA
- OpenMP

#Compiling
```
cd gpuvmem
./configure
make
```
#Usage

Convert your ms input file to sql with ms2sql:

`ms2sql --ms-src input.ms --db-dest input.sql --verbose`

Remember to also make a copy of the input.sql to save residuals.

`cp input.sql residuals.sql`

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
- -h  --help       Shows this
- -i  --input      The name of the input file of visibilities(SQLite)
- -o  --output     The name of the output file of residual visibilities(SQLite)
- -d  --inputdat   The name of the input file of parameters
- -m  --modin      mod_in_0 FITS file
- -b  --beam       beam_0 FITS file
- -g  --multigpu   Number of GPUs to use multiGPU image synthesis (Default OFF => 0)
- -s  --select     If multigpu option is OFF, then select the GPU ID of the GPU you will work on. (Default = 0)

An out folder will be created. That's where outputs (FITS images) will be saved.

#Contributors

- Miguel Cárcamo
- Fernando Rannou
- Pablo Román
- Simón Casassus
- Victor Moral
