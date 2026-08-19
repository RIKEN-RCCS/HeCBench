FC ?= nvfortran
GPU_ARCH ?= cc90
FFLAGS ?= -O3 -mp=gpu -gpu=$(GPU_ARCH) -Mextend
LAUNCHER ?=
RUN_ARGS ?= -i 1000 -l 32 -t 128 -v 3 -w 1
PROGRAM := main
SOURCES := su3.f90 lattice.f90 mat_nn_openmp.f90 main.f90

.PHONY: all clean run
all: $(PROGRAM)

$(PROGRAM): $(SOURCES)
	$(FC) $(FFLAGS) -o $@ $(SOURCES)

run: $(PROGRAM)
	OMP_TARGET_OFFLOAD=MANDATORY $(LAUNCHER) ./$(PROGRAM) -i 1000 -l 32 -t 128 -v 3 -w 1
clean:
	rm -f $(PROGRAM) *.o *.mod
