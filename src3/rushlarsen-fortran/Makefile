FC ?= nvfortran
GPU_ARCH ?= cc90
FFLAGS ?= -O3 -mp=gpu -gpu=$(GPU_ARCH) -Mextend
PROGRAM := main
SOURCES := utils.f90 kernels.f90 reference.f90 main.f90
LAUNCHER ?=

.PHONY: all run clean

all: $(PROGRAM)

$(PROGRAM): $(SOURCES)
	$(FC) $(FFLAGS) -o $@ $(SOURCES)

run: $(PROGRAM)
	OMP_TARGET_OFFLOAD=MANDATORY $(LAUNCHER) ./$(PROGRAM) 100 100000
clean:
	rm -f $(PROGRAM) *.o *.mod
