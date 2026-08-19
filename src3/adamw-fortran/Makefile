FC       ?= nvfortran
GPU_ARCH ?= cc90
FFLAGS   ?= -O3 -mp=gpu -gpu=$(GPU_ARCH)
PROGRAM  := main
SOURCES  := adamw_kernels.f90 adamw_reference.f90 main.f90
LAUNCHER ?=
RUN_ARGS ?= 1000000 100

.PHONY: all clean run

all: $(PROGRAM)

$(PROGRAM): $(SOURCES)
	$(FC) $(FFLAGS) -o $@ $(SOURCES)

run: $(PROGRAM)
	OMP_TARGET_OFFLOAD=MANDATORY $(LAUNCHER) ./$(PROGRAM) $(RUN_ARGS)

clean:
	rm -f $(PROGRAM) *.o *.mod
