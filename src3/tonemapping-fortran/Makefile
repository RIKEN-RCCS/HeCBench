FC ?= nvfortran
GPU_ARCH ?= cc90
FFLAGS ?= -O3 -mp=gpu -gpu=$(GPU_ARCH) -Mextend
LAUNCHER ?=
RUN_ARGS ?= ../tonemapping-sycl/input.hdr 10000
PROGRAM := main
SOURCES := main.f90

.PHONY: all clean run
all: $(PROGRAM)

$(PROGRAM): $(SOURCES)
	$(FC) $(FFLAGS) -o $@ $(SOURCES)

run: $(PROGRAM)
	OMP_TARGET_OFFLOAD=MANDATORY $(LAUNCHER) ./$(PROGRAM) ../tonemapping-sycl/input.hdr 10000
clean:
	rm -f $(PROGRAM) *.o *.mod
