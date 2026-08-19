FC ?= nvfortran
GPU_ARCH ?= cc90
FFLAGS ?= -O3 -mp=gpu -gpu=$(GPU_ARCH) -Mextend
LAUNCHER ?=
RUN_ARGS ?= ../svd3x3-cuda/Dataset_1M.txt 1000
PROGRAM := main
SOURCES := kernels.f90 main.f90

.PHONY: all clean run
all: $(PROGRAM)

$(PROGRAM): $(SOURCES)
	$(FC) $(FFLAGS) -o $@ $(SOURCES)

run: $(PROGRAM)
	OMP_TARGET_OFFLOAD=MANDATORY $(LAUNCHER) ./$(PROGRAM) ../svd3x3-cuda/Dataset_1M.txt 1000
clean:
	rm -f $(PROGRAM) *.o *.mod
