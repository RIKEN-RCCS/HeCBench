FC ?= nvfortran
GPU_ARCH ?= cc90
FFLAGS ?= -O3 -mp=gpu -gpu=$(GPU_ARCH) -Mextend
PROGRAM := main
SOURCES := main.f90
LAUNCHER ?=
RUN_ARGS ?= 1024 1024 0.001 1000
.PHONY: all clean run
all: $(PROGRAM)
$(PROGRAM): $(SOURCES)
	$(FC) $(FFLAGS) -o $@ $(SOURCES)
run: $(PROGRAM)
	OMP_TARGET_OFFLOAD=MANDATORY $(LAUNCHER) ./$(PROGRAM) $(RUN_ARGS)
clean:
	rm -f $(PROGRAM) *.o *.mod
