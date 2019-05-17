
CUFFTFLAG += -lcufft
CFLAGS += -D_FORCE_INLINES -c -w -O3 -Xptxas -O3
INC_DIRS += -Iinclude -I/usr/local/include/casacore/
CFFLAG += -lcfitsio -lm -lcasa_casa -lcasa_tables -lcasa_ms -lcasa_measures
LDFLAGS += -lcuda -lcudart
FOPENFLAG += -Xcompiler -fopenmp -lgomp
CCFLAG += -lstdc++
# Gencode arguments
SMS ?= 30 35 37 50 52

ifeq ($(NEWCASA),1)
CFLAGS += -DNEWCASA
endif

ifeq ($(SMS),)
$(info >>> WARNING - no SM architectures have been specified <<<)
endif

ifeq ($(ARCHFLAG),)
# Generate SASS code for each SM architecture listed in $(SMS)
$(foreach sm,$(SMS),$(eval ARCHFLAG += -gencode arch=compute_$(sm),code=sm_$(sm)))

# Generate PTX code from the highest SM architecture in $(SMS) to guarantee forward-compatibility
HIGHEST_SM := $(lastword $(sort $(SMS)))
ifneq ($(HIGHEST_SM),)
ARCHFLAG += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)
endif
endif

main:	build/main.o build/MSFITSIO.o build/functions.o build/directioncosines.o build/copyrightwarranty.o build/rngs.o build/rvgs.o build/f1dim.o build/mnbrak.o build/brent.o build/linmin.o  build/frprmn.o build/synthesizer.o build/imageProcessor.o build/chi2.o build/laplacian.o build/gridding.o build/entropy.o build/ioms.o build/totalvariation.o build/quadraticpenalization.o build/error.o build/lbfgs.o
	@ echo "Linking CUDAMEM"
	@ mkdir -p bin
	@ nvcc build/*.o -o bin/gpuvmem $(LDFLAGS) $(CFFLAG) $(FOPENFLAG) $(CUFFTFLAG) $(ARCHFLAG) $(CCFLAG)
	@ echo "The compilation has been completed successfully"

build/main.o: src/main.cu
	@ echo "Building Main"
	@ mkdir -p build
	@ nvcc $(CFLAGS) $(INC_DIRS) src/main.cu -o build/main.o $(LDFLAGS) $(CFFLAG) $(FOPENFLAG) $(CUFFTFLAG) $(ARCHFLAG)

build/chi2.o: src/chi2.cu
	@ echo "Building chi2"
	@ mkdir -p build
	@ nvcc $(CFLAGS) $(INC_DIRS) src/chi2.cu -o build/chi2.o $(LDFLAGS) $(CFFLAG) $(FOPENFLAG) $(CUFFTFLAG) $(ARCHFLAG)

build/laplacian.o: src/laplacian.cu
	@ echo "Building laplacian"
	@ mkdir -p build
	@ nvcc $(CFLAGS) $(INC_DIRS) src/laplacian.cu -o build/laplacian.o $(LDFLAGS) $(CFFLAG) $(FOPENFLAG) $(CUFFTFLAG) $(ARCHFLAG)

build/entropy.o: src/entropy.cu
	@ echo "Building entropy"
	@ mkdir -p build
	@ nvcc $(CFLAGS) $(INC_DIRS) src/entropy.cu -o build/entropy.o $(LDFLAGS) $(CFFLAG) $(FOPENFLAG) $(CUFFTFLAG) $(ARCHFLAG)

build/quadraticpenalization.o: src/quadraticpenalization.cu
	@ echo "Building quadratic penalization"
	@ mkdir -p build
	@ nvcc $(CFLAGS) $(INC_DIRS) src/quadraticpenalization.cu -o build/quadraticpenalization.o $(LDFLAGS) $(CFFLAG) $(FOPENFLAG) $(CUFFTFLAG) $(ARCHFLAG)

build/totalvariation.o: src/totalvariation.cu
	@ echo "Building Total Variation"
	@ mkdir -p build
	@ nvcc $(CFLAGS) $(INC_DIRS) src/totalvariation.cu -o build/totalvariation.o $(LDFLAGS) $(CFFLAG) $(FOPENFLAG) $(CUFFTFLAG) $(ARCHFLAG)

build/ioms.o: src/ioms.cu
	@ echo "Building ioms"
	@ mkdir -p build
	@ nvcc $(CFLAGS) $(INC_DIRS) src/ioms.cu -o build/ioms.o $(LDFLAGS) $(CFFLAG) $(FOPENFLAG) $(CUFFTFLAG) $(ARCHFLAG)

build/copyrightwarranty.o: src/copyrightwarranty.cu
	@ echo "Building copyrightwarranty"
	@ mkdir -p build
	@ nvcc $(CFLAGS) $(INC_DIRS) src/copyrightwarranty.cu -o build/copyrightwarranty.o $(LDFLAGS) $(CFFLAG) $(FOPENFLAG) $(CUFFTFLAG) $(ARCHFLAG)

build/imageProcessor.o: src/imageProcessor.cu
	@ echo "Building imageProcessor"
	@ mkdir -p build
	@ nvcc $(CFLAGS) $(INC_DIRS) src/imageProcessor.cu -o build/imageProcessor.o $(LDFLAGS) $(CFFLAG) $(FOPENFLAG) $(CUFFTFLAG) $(ARCHFLAG)

build/synthesizer.o: src/synthesizer.cu
	@ echo "Building synthesizer"
	@ mkdir -p build
	@ nvcc $(CFLAGS) $(INC_DIRS) src/synthesizer.cu -o build/synthesizer.o $(LDFLAGS) $(CFFLAG) $(FOPENFLAG) $(CUFFTFLAG) $(ARCHFLAG)

build/gridding.o: src/gridding.cu
	@ echo "Building gridding"
	@ mkdir -p build
	@ nvcc $(CFLAGS) $(INC_DIRS) src/gridding.cu -o build/gridding.o $(LDFLAGS) $(CFFLAG) $(FOPENFLAG) $(CUFFTFLAG) $(ARCHFLAG)

build/error.o: src/error.cu
	@ echo "Building Error"
	@ mkdir -p build
	@ nvcc $(CFLAGS) $(INC_DIRS) src/error.cu -o build/error.o $(LDFLAGS) $(CFFLAG) $(FOPENFLAG) $(CUFFTFLAG) $(ARCHFLAG)

build/MSFITSIO.o: src/MSFITSIO.cu
	@ echo "Building MSFITSIO"
	@ nvcc $(CFLAGS) $(INC_DIRS) src/MSFITSIO.cu -o build/MSFITSIO.o $(LDFLAGS) $(CFFLAG) $(FOPENFLAG) $(CUFFTFLAG) $(ARCHFLAG)

build/functions.o: src/functions.cu
	@ echo "Building Functions"
	@ nvcc $(CFLAGS) $(INC_DIRS) src/functions.cu -o build/functions.o $(LDFLAGS) $(CFFLAG) $(FOPENFLAG) $(CUFFTFLAG) $(ARCHFLAG)

build/directioncosines.o: src/directioncosines.cu
	@ echo "Building directioncosines"
	@ nvcc $(CFLAGS) $(INC_DIRS) src/directioncosines.cu -o build/directioncosines.o  $(LDFLAGS) $(CFFLAG) $(ARCHFLAG)

build/rngs.o: src/rngs.cu
	@ echo "Building Random number generator"
	@ nvcc $(CFLAGS) $(INC_DIRS) src/rngs.cu -o build/rngs.o $(LDFLAGS) $(CFFLAG) $(ARCHFLAG)

build/rvgs.o: src/rvgs.cu
	@ echo "Building Random number generator 2"
	@ nvcc $(CFLAGS) $(INC_DIRS) src/rvgs.cu -o build/rvgs.o $(LDFLAGS) $(CFFLAG) $(ARCHFLAG)

build/f1dim.o: src/f1dim.cu
	@ echo "Building f1dim"
	@ nvcc $(CFLAGS) $(INC_DIRS) src/f1dim.cu -o build/f1dim.o $(LDFLAGS) $(CFFLAG) $(ARCHFLAG)

build/mnbrak.o: src/mnbrak.cu
	@ echo "Building mnbrak"
	@ nvcc $(CFLAGS) $(INC_DIRS) src/mnbrak.cu -o build/mnbrak.o $(LDFLAGS) $(CFFLAG) $(ARCHFLAG)

build/brent.o: src/brent.cu
	@ echo "Building brent"
	@ nvcc $(CFLAGS) $(INC_DIRS) src/brent.cu -o build/brent.o $(LDFLAGS) $(CFFLAG) $(ARCHFLAG)

build/linmin.o: src/linmin.cu
	@ echo "Building linmin"
	@ nvcc $(CFLAGS) $(INC_DIRS) src/linmin.cu -o build/linmin.o $(LDFLAGS) $(CFFLAG) $(ARCHFLAG)

build/frprmn.o: src/frprmn.cu
	@ echo "Building frprmn"
	@ nvcc $(CFLAGS) $(INC_DIRS) src/frprmn.cu -o build/frprmn.o $(LDFLAGS) $(CFFLAG) $(ARCHFLAG)

build/lbfgs.o: src/lbfgs.cu
	@ echo "Building lbfgs"
	@ nvcc $(CFLAGS) $(INC_DIRS) src/lbfgs.cu -o build/lbfgs.o $(LDFLAGS) $(CFFLAG) $(ARCHFLAG)

cleanall:
	@ echo "Cleaning all folders.."
	@ rm -rf build/*
	@ rm -rf bin/*
	@ rm -r *.fits

clean:
	@ echo "Cleaning gpuvmem folders.."
	@ rm -rf build/*
	@ rm -rf bin/*

co65:
	@ ./bin/gpuvmem -i ./tests/co65/co65.ms -o ./tests/co65/co65_out.ms -O ./tests/co65/mod_out.fits -m ./tests/co65/mod_in_0.fits -I ./tests/co65/input.dat -p ./tests/co65/mem/ -X 16 -Y 16 -V 256 -z 0.001 -Z 0.01 -g 2 -R 2.0 -t 500000000 -g 2 --verbose
selfcalband9:
	@ cuda-memcheck ./bin/gpuvmem -i ./tests/selfcalband9/hd142_b9cont_self_tav.ms -o ./tests/selfcalband9/hd142_b9cont_out.ms -O ./tests/selfcalband9/mod_out.fits -m ./tests/selfcalband9/mod_in_0.fits -I ./tests/selfcalband9/input.dat -p ./tests/selfcalband9/mem/ -X 8 -Y 8 -V 256 --verbose -z 0.001 -Z 0.05,0.5 -t 500000000 --print-images
freq78:
	@ ./bin/gpuvmem -i ./tests/FREQ78/FREQ78.ms -o ./tests/FREQ78/FREQ78_out.ms -O ./tests/FREQ78/mod_out.fits -m ./tests/FREQ78/mod_in_0.fits -I ./tests/FREQ78/input.dat -p ./tests/FREQ78/mem/ -X 16 -Y 16 -V 256 -z 0.001 -Z 0.005 -t 500000000 -g 2 -R 0.0 --verbose --print-images
antennae:
	@ ./bin/gpuvmem -i ./tests/antennae/all_fields.ms -o ./tests/antennae/antennae_out.ms -O ./tests/antennae/mod_out.fits -m ./tests/antennae/mod_in_0.fits -I ./tests/antennae/input.dat -p ./tests/antennae/mem/ -X 16 -Y 16 -V 256  -z 0.001 -Z 0.05 -t 500000000 --verbose --print-images
