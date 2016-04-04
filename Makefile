# ALMA Image Reconstruction MEM
# by Miguel Cárcamo

CUFFTFLAG = -lcufft
CFLAGS = -c -w
INC_DIRS = -Iinclude
CFFLAG = -Llib -lcfitsio -lm
SQLITE = -lsqlite3
LDFLAGS = -lcuda -lcudart
ARCHFLAG = -arch=sm_50
FOPENFLAG = -Xcompiler -fopenmp -lgomp

main: build/main.o build/functions.o build/directioncosines.o build/rngs.o build/f1dim.o build/mnbrak.o build/brent.o build/linmin.o build/frprmn.o
	@ echo "Linking CUDAMEM"
	@ mkdir -p bin
	@ nvcc $(LDFLAGS) build/*.o -o bin/mem $(CFFLAG) $(FOPENFLAG) $(SQLITE) $(CUFFTFLAG) $(ARCHFLAG)
	@ echo "The compilation has been completed successfully"

build/main.o: src/main.cu
	@ echo "Building Main"
	@ mkdir -p build
	@ nvcc $(CFLAGS) $(INC_DIRS) $(LDFLAGS) src/main.cu -o build/main.o $(CFFLAG) $(FOPENFLAG) $(CUFFTFLAG) $(ARCHFLAG)

build/functions.o: src/functions.cu
	@ echo "Building Functions"
	@ mkdir -p lib
	@ cd cfitsio; make; cp libcfitsio.a ../lib/.
	@ nvcc $(CFLAGS) $(INC_DIRS) $(LDFLAGS) src/functions.cu -o build/functions.o $(CFFLAG) $(FOPENFLAG) $(SQLITE) $(CUFFTFLAG) $(ARCHFLAG)

build/directioncosines.o: src/directioncosines.cu
	@ echo "Building directioncosines"
	@ nvcc $(CFLAGS) $(INC_DIRS) $(LDFLAGS) src/directioncosines.cu -o build/directioncosines.o $(CFFLAG) $(ARCHFLAG)

build/rngs.o: src/rngs.cu
	@ echo "Building Random number generator"
	@ nvcc $(CFLAGS) $(INC_DIRS) $(LDFLAGS) src/rngs.cu -o build/rngs.o $(CFFLAG) $(ARCHFLAG)

build/f1dim.o: src/f1dim.cu
	@ echo "Building f1dim"
	@ nvcc $(CFLAGS) $(INC_DIRS) $(LDFLAGS) src/f1dim.cu -o build/f1dim.o $(CFFLAG) $(ARCHFLAG)

build/mnbrak.o: src/mnbrak.cu
	@ echo "Building mnbrak"
	@ nvcc $(CFLAGS) $(INC_DIRS) $(LDFLAGS) src/mnbrak.cu -o build/mnbrak.o $(CFFLAG) $(ARCHFLAG)

build/brent.o: src/brent.cu
	@ echo "Building brent"
	@ nvcc $(CFLAGS) $(INC_DIRS) $(LDFLAGS) src/brent.cu -o build/brent.o $(CFFLAG) $(ARCHFLAG)

build/linmin.o: src/linmin.cu
	@ echo "Building linmin"
	@ nvcc $(CFLAGS) $(INC_DIRS) $(LDFLAGS) src/linmin.cu -o build/linmin.o $(CFFLAG) $(ARCHFLAG)

build/frprmn.o: src/frprmn.cu
	@ echo "Building frprmn"
	@ nvcc $(CFLAGS) $(INC_DIRS) $(LDFLAGS) src/frprmn.cu -o build/frprmn.o $(CFFLAG) $(ARCHFLAG)

clean:
	@ clear
	@ echo "Cleaning folders.."
	@ rm -rf build/*
	@ rm -rf bin/*
	@ rm -f lib/*.a
	@ cd cfitsio; make clean
	@ cd cfitsio; make distclean

conf:
	@ clear
	@ echo "Doing configure..."
	@ ./configure

co65:
	@ clear
	@ ./bin/mem -i ./tests/co65/co65.sqlite -o ./tests/co65/co65_out.sqlite -m ./tests/co65/mod_in_0.fits -b ./tests/co65/beam_0.fits -d ./tests/co65/input.dat -p ./tests/co65/mem/

selfcalband9:
	@ clear
	@ ./bin/mem -i ./tests/selfcalband9/selfcalband9.sqlite -o ./tests/selfcalband9/selfcalband9_out.sqlite -m ./tests/selfcalband9/mod_in_0.fits -b ./tests/selfcalband9/beam_0.fits -d ./tests/selfcalband9/input.dat -p ./tests/selfcalband9/mem/

hco20:
	@ clear
	@ ./bin/mem -i ./tests/HCO-chan20/HCO-20.sqlite -o ./tests/HCO-chan20/HCO-20_out.sqlite -m ./tests/HCO-chan20/mod_in_0.fits -b ./tests/HCO-chan20/beam_0.fits -d ./tests/HCO-chan20/input.dat -p ./tests/HCO-chan20/mem/

hco23:
	@ clear
	@ ./bin/mem -i ./tests/HCO-chan23/HCO-23.sqlite -o ./tests/HCO-chan23/HCO-23_out.sqlite -m ./tests/HCO-chan23/mod_in_0.fits -b ./tests/HCO-chan23/beam_0.fits -d ./tests/HCO-chan23/input.dat -p ./tests/HCO-chan23/mem/

freq78:
	@ clear
	@ ./bin/mem -i ./tests/FREQ78/FREQ78.sqlite -o ./tests/FREQ78/FREQ78_out.sqlite -m ./tests/FREQ78/mod_in_0.fits -b ./tests/FREQ78/beam_0.fits -d ./tests/FREQ78/input.dat -p ./tests/FREQ78/mem/

phantom:
	@ clear
	@ ./bin/mem -i ./tests/phantom/phantom.sql -o ./tests/phantom/phantom_out.sql -m ./tests/phantom/mod_in_0.fits -b ./tests/phantom/beam_0.fits -d ./tests/phantom/input.dat -p ./tests/phantom/mem/
